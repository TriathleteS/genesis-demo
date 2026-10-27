"""ヒューマノイド Unitree G1（脚 12 関節）の強化学習用の環境。

Genesis 公式の Go2 環境（go2_env.py）と同じ構成で書き、観測・報酬・終了条件は
Unitree 公式の学習コード unitree_rl_gym（Isaac Gym 用、BSD 3-Clause License）の
G1 の設定を Genesis 向けに移植したもの。
https://github.com/unitreerobotics/unitree_rl_gym/tree/main/legged_gym/envs/g1

Go2 との主な違い:
  - 歩行のリズムを与える「位相」（周期 0.8 秒）を観測に含め、左右の足を交互に接地させる報酬を使う
  - 前後・左右・旋回の速度指令にランダムに従うよう学習する（Go2 は前進 0.5 m/s のみ）
  - 胴体を外から押す・床の摩擦を変えるなどのランダム化で、転びにくい歩き方を学ばせる
  - 方策（actor）には見えない胴体の速度を、価値関数（critic）にだけ与える
"""

import math
import os

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

# リポジトリ内に同梱した G1 のモデル（assets/unitree_g1/g1_12dof.urdf）
DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "unitree_g1", "g1_12dof.urdf"
)


def gs_rand(lower, upper, batch_shape):
    lower = torch.as_tensor(lower, dtype=gs.tc_float, device=gs.device)
    upper = torch.as_tensor(upper, dtype=gs.tc_float, device=gs.device)
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class G1Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = env_cfg["dt"]  # 方策が行動を決める周期（0.02 秒 = 50 Hz）
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = env_cfg  # rsl_rl の OnPolicyRunner が参照する
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = dict(reward_cfg["reward_scales"])

        # ------------------------------------------------------------------
        # シーンの作成
        # ------------------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=env_cfg["substeps"],  # 物理計算は dt / substeps = 0.005 秒刻み（Unitree と同じ）
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # 自己衝突を無効にしているので、衝突ペアは「ロボットの各形状 × 床」だけ。
                # G1 の衝突形状は 32 個なので、それを上回る値にしておく（超えると例外になる）
                max_collision_pairs=env_cfg["max_collision_pairs"],
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],  # ビューアーには 1 体目だけを描画する
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, -2.0, 1.2),
                camera_lookat=(0.0, 0.0, 0.7),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=env_cfg.get("urdf_path") or DEFAULT_URDF,
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        self.scene.build(n_envs=num_envs)

        # ------------------------------------------------------------------
        # 関節・リンクの対応付けと PD ゲイン
        # ------------------------------------------------------------------
        joint_names = env_cfg["joint_names"]
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in joint_names], dtype=gs.tc_int, device=gs.device
        )

        def per_joint(table):
            # {"hip_yaw": 100, "knee": 150, ...} のような部分一致の表から、関節ごとの値を作る
            values = []
            for name in joint_names:
                matches = [v for key, v in table.items() if key in name]
                assert len(matches) == 1, f"{name} に対応する値が見つからない、または複数ある"
                values.append(matches[0])
            return values

        self.robot.set_dofs_kp(per_joint(env_cfg["stiffness"]), self.motors_dof_idx)
        self.robot.set_dofs_kv(per_joint(env_cfg["damping"]), self.motors_dof_idx)

        # 関節の可動範囲（報酬で範囲の端に近づくのを抑えるため、少し内側を「ソフトな上限」にする）
        lower, upper = self.robot.get_dofs_limit(self.motors_dof_idx)
        lower, upper = lower.reshape(-1, self.num_actions)[0], upper.reshape(-1, self.num_actions)[0]
        mid, half = (lower + upper) / 2, (upper - lower) / 2 * reward_cfg["soft_dof_pos_limit"]
        self.soft_dof_limits = (mid - half, mid + half)

        self.feet_links_idx = [self.robot.get_link(name).idx_local for name in env_cfg["feet_link_names"]]
        self.termination_links_idx = [
            self.robot.get_link(name).idx_local for name in env_cfg["termination_contact_link_names"]
        ]
        self.hip_dofs_idx = [joint_names.index(name) for name in env_cfg["hip_joint_names"]]

        # 床との摩擦を環境ごとにランダムに変える（滑りやすい床・滑りにくい床の両方で歩けるようにする）
        if env_cfg["randomize_friction"]:
            ratio = gs_rand(*env_cfg["friction_range"], (num_envs, 1)).repeat(1, self.robot.n_links)
            self.robot.set_friction_ratio(ratio, links_idx_local=list(range(self.robot.n_links)))

        # ------------------------------------------------------------------
        # 初期状態とバッファ
        # ------------------------------------------------------------------
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in joint_names], dtype=gs.tc_float, device=gs.device
        )
        init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, inv_quat(self.init_base_quat))

        n, a = num_envs, self.num_actions
        f = dict(dtype=gs.tc_float, device=gs.device)
        self.base_pos = torch.zeros((n, 3), **f)
        self.base_quat = torch.zeros((n, 4), **f)
        self.base_euler = torch.zeros((n, 3), **f)
        self.base_lin_vel = torch.zeros((n, 3), **f)
        self.base_ang_vel = torch.zeros((n, 3), **f)
        self.projected_gravity = torch.zeros((n, 3), **f)
        self.dof_pos = torch.zeros((n, a), **f)
        self.dof_vel = torch.zeros((n, a), **f)
        self.last_dof_vel = torch.zeros((n, a), **f)
        self.actions = torch.zeros((n, a), **f)
        self.last_actions = torch.zeros((n, a), **f)
        self.feet_pos = torch.zeros((n, len(self.feet_links_idx), 3), **f)
        self.feet_vel = torch.zeros((n, len(self.feet_links_idx), 3), **f)
        self.contact_forces = torch.zeros((n, self.robot.n_links, 3), **f)
        self.leg_phase = torch.zeros((n, 2), **f)
        self.phase = torch.zeros((n,), **f)
        self.rew_buf = torch.zeros((n,), **f)
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=gs.device)
        self.time_out_buf = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((n, self.num_commands), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f
        )
        self.commands_limits = tuple(
            torch.tensor(values, **f)
            for values in zip(
                command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]
            )
        )
        self.fixed_command = None  # 評価時に速度指令を固定するためのもの（None なら学習時と同じくランダム）
        self.common_step_counter = 0
        self.extras = dict()

        # 観測ノイズの大きさ（実機のセンサーノイズを模擬して、方策を頑健にする）
        noise = obs_cfg["noise_scales"]
        level = obs_cfg["noise_level"] if obs_cfg["add_noise"] else 0.0
        self.noise_scale_vec = torch.cat(
            (
                torch.full((3,), noise["ang_vel"] * self.obs_scales["ang_vel"]),
                torch.full((3,), noise["gravity"]),
                torch.zeros(3),  # 速度指令
                torch.full((a,), noise["dof_pos"] * self.obs_scales["dof_pos"]),
                torch.full((a,), noise["dof_vel"] * self.obs_scales["dof_vel"]),
                torch.zeros(a),  # 前回の行動
                torch.zeros(2),  # 位相
            )
        ).to(**f) * level

        # 報酬関数の準備（係数には dt を掛けておく）
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((n,), **f)

        self.reset()

    # ----------------------------------------------------------------------
    # 速度指令
    # ----------------------------------------------------------------------
    def _resample_commands(self, envs_idx):
        if self.fixed_command is not None:
            self.commands[:] = self.fixed_command
            return
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # ごく小さい並進指令は 0 にして、「その場で足踏みして立ち止まる」ことも学習させる
        commands[:, :2] *= (torch.norm(commands[:, :2], dim=1) > 0.2).unsqueeze(1)
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def set_command(self, vx, vy, wz):
        """評価用：速度指令を固定する（前後 vx [m/s], 左右 vy [m/s], 旋回 wz [rad/s]）。"""
        self.fixed_command = torch.tensor([vx, vy, wz], dtype=gs.tc_float, device=gs.device)
        self.commands[:] = self.fixed_command

    # ----------------------------------------------------------------------
    # 1 ステップ進める
    # ----------------------------------------------------------------------
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 状態の更新
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(self.base_quat, rpy=True, degrees=False)  # [roll, pitch, yaw] [rad]
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)  # 胴体座標系の速度
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)  # 胴体の傾き
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.feet_pos = self.robot.get_links_pos(self.feet_links_idx)
        self.feet_vel = self.robot.get_links_vel(self.feet_links_idx)
        self.contact_forces = self.robot.get_links_net_contact_force()

        # 歩行の位相（0〜1 を周期的に繰り返す）。左右の足は半周期ずらす
        self.phase = (self.episode_length_buf * self.dt) % self.env_cfg["gait_period"] / self.env_cfg["gait_period"]
        self.leg_phase = torch.stack((self.phase, (self.phase + 0.5) % 1.0), dim=1)

        # 速度指令の更新
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # 終了判定（転倒・胴体の接地・時間切れ）
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= torch.any(
            torch.norm(self.contact_forces[:, self.termination_links_idx], dim=-1) > 1.0, dim=1
        )
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # 報酬の計算
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            # 合計が負にならないようにする（学習初期に「早く転んで罰を減らす」方向へ逃げるのを防ぐ）
            self.rew_buf.clip_(min=0.0)

        self.extras["time_outs"] = self.time_out_buf.to(dtype=gs.tc_float)

        # 終了した環境をリセット
        self._reset_idx(self.reset_buf)

        # 一定間隔で胴体を横から押す（外乱に強い歩き方を学ばせる）
        if self.env_cfg["push_robots"] and self.common_step_counter % int(self.env_cfg["push_interval_s"] / self.dt) == 0:
            max_vel = self.env_cfg["max_push_vel_xy"]
            push_vel = gs_rand([-max_vel, -max_vel], [max_vel, max_vel], (self.num_envs,))
            self.robot.set_dofs_velocity(push_vel, dofs_idx_local=[0, 1])

        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf, "critic": self.privileged_obs_buf}, batch_size=[self.num_envs])

    def _update_observation(self):
        phase_obs = torch.stack((torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)), dim=1)
        common = (
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
            phase_obs,  # 2
        )
        ang_vel = self.base_ang_vel * self.obs_scales["ang_vel"]  # 3
        # 方策への観測（47 次元）。実機で測れる量だけを使い、ノイズを加える
        self.obs_buf = torch.cat((ang_vel, *common), dim=-1)
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # 価値関数への観測（50 次元）。実機では直接測りにくい胴体の速度も含める
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales["lin_vel"], ang_vel, *common), dim=-1)

    # ----------------------------------------------------------------------
    # リセット
    # ----------------------------------------------------------------------
    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                        self.last_dof_vel, self.episode_length_buf):
                buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                        self.last_dof_vel):
                buf.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # 学習ログ用：終了したエピソードの報酬の内訳（1 秒あたり）
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    def reset(self):
        self._reset_idx()
        self.phase.zero_()
        self.leg_phase = torch.stack((self.phase, (self.phase + 0.5) % 1.0), dim=1)
        self._update_observation()
        return self.get_observations()

    # ----------------------------------------------------------------------
    # 報酬関数（unitree_rl_gym の legged_robot.py / g1_env.py と同じ定義）
    # ----------------------------------------------------------------------
    def _feet_in_contact(self):
        return torch.norm(self.contact_forces[:, self.feet_links_idx], dim=-1) > 1.0

    def _reward_tracking_lin_vel(self):
        # 前後・左右の速度指令に従う
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # 旋回の速度指令に従う
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # 胴体の上下動を抑える
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # 胴体の横揺れ・縦揺れを抑える
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # 胴体を水平に保つ
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # 胴体の高さを目標値に保つ
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_dof_acc(self):
        # 関節の急加速を抑える
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        # 関節の速すぎる動きを抑える
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        # 行動の急変を抑える（動きを滑らかにする）
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_pos_limits(self):
        # 関節の可動範囲の端に近づくのを抑える
        lower, upper = self.soft_dof_limits
        out_of_limits = -(self.dof_pos - lower).clip(max=0.0) + (self.dof_pos - upper).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_alive(self):
        # 転ばずにいること自体への報酬
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _reward_hip_pos(self):
        # 股関節を左右に開いたり捻ったりしすぎないようにする（hip_roll, hip_yaw）
        return torch.sum(torch.square(self.dof_pos[:, self.hip_dofs_idx]), dim=1)

    def _reward_contact_no_vel(self):
        # 接地している足が滑らないようにする
        contact = self._feet_in_contact()
        return torch.sum(torch.square(self.feet_vel * contact.unsqueeze(-1)), dim=(1, 2))

    def _reward_feet_swing_height(self):
        # 遊脚（浮いている足）を目標の高さまで上げる
        contact = self._feet_in_contact()
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.reward_cfg["feet_swing_height_target"]) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_contact(self):
        # 位相が立脚期のときは接地、遊脚期のときは離地していれば報酬（左右交互のリズムを作る）
        is_stance = self.leg_phase < self.env_cfg["stance_ratio"]
        contact = self.contact_forces[:, self.feet_links_idx, 2] > 1.0
        return torch.sum((~(contact ^ is_stance)).to(dtype=gs.tc_float), dim=1)
