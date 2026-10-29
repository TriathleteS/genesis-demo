"""ヒューマノイド Unitree G1（23 関節：脚・腰・腕）が、正面の台によじ登る強化学習の環境。

歩行（g1_env.py）との主な違い:
  - 目標到達型のタスク：台に近づき、手をつき、体を引き上げ、台の上に立つ（周期的な歩行の位相は使わない）
  - 進み具合の報酬は「これまでの最高到達度を更新したときだけ」与える（APEX と同じ考え方。
    行ったり来たりして報酬を稼ぐ抜け道を防ぐ）。到達度は主に「全身の重心がどれだけ上がったか」で測り、
    手か足が台の上に 0.1 秒以上触れている間だけ数える（跳び上がった一瞬の高さを数えないため）
  - 台の高さのカリキュラム：環境ごとに台の高さの段階を持ち、登れたら高く、登れなければ低くする
  - 補助の力のカリキュラム：学習の初期は、台の近くで腰を持ち上げる力をかけ、学習が進むにつれて 0 にする
    （参照動作なしで「体を引き上げる」動きを見つけさせるため）

座標：台の正面の面が x = 0。ロボットは x < 0 から +x 方向（台の方）を向いて始める。
"""

import math
import os

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, xyz_to_quat

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "unitree_g1", "g1_23dof_rev_1_0.urdf"
)


def gs_rand(lower, upper, batch_shape):
    lower = torch.as_tensor(lower, dtype=gs.tc_float, device=gs.device)
    upper = torch.as_tensor(upper, dtype=gs.tc_float, device=gs.device)
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class G1ClimbEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, curriculum_cfg, show_viewer=False, camera_res=None):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = env_cfg  # rsl_rl の OnPolicyRunner が参照する
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.curriculum_cfg = curriculum_cfg
        self.obs_scales = obs_cfg["obs_scales"]

        # ------------------------------------------------------------------
        # シーン：床・台・ロボット
        # ------------------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=env_cfg["self_collision"],
                max_collision_pairs=env_cfg["max_collision_pairs"],
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-1.2, -2.6, 1.6),
                camera_lookat=(0.0, 0.0, 0.6),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # 台は高さ box_size[2] の直方体を床に埋め、上面の高さが目標の値になるよう上下に動かして使う
        # （形状は全環境で共通だが、位置は環境ごとに変えられるため）
        self.box_size = env_cfg["box_size"]
        self.box = self.scene.add_entity(
            gs.morphs.Box(
                pos=(self.box_size[0] / 2, 0.0, -self.box_size[2] / 2),
                size=self.box_size,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.55, 0.45, 0.35)),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=env_cfg.get("urdf_path") or DEFAULT_URDF, pos=(-0.5, 0.0, 0.8)),
        )

        self.camera = None
        if camera_res is not None:
            self.camera = self.scene.add_camera(
                res=camera_res, pos=(-1.2, -2.6, 1.6), lookat=(0.0, 0.0, 0.6), fov=40, GUI=False, env_idx=0
            )

        self.scene.build(n_envs=num_envs)

        # ------------------------------------------------------------------
        # 関節・リンク
        # ------------------------------------------------------------------
        joint_names = env_cfg["joint_names"]
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in joint_names], dtype=gs.tc_int, device=gs.device
        )

        def per_joint(table):
            values = []
            for name in joint_names:
                matches = [v for key, v in table.items() if key in name]
                assert len(matches) == 1, f"{name} に対応する値が見つからない、または複数ある"
                values.append(matches[0])
            return values

        self.robot.set_dofs_kp(per_joint(env_cfg["stiffness"]), self.motors_dof_idx)
        self.robot.set_dofs_kv(per_joint(env_cfg["damping"]), self.motors_dof_idx)
        self.action_scale = torch.tensor(per_joint(env_cfg["action_scale"]), dtype=gs.tc_float, device=gs.device)

        lower, upper = self.robot.get_dofs_limit(self.motors_dof_idx)
        lower, upper = lower.reshape(-1, self.num_actions)[0], upper.reshape(-1, self.num_actions)[0]
        mid, half = (lower + upper) / 2, (upper - lower) / 2 * reward_cfg["soft_dof_pos_limit"]
        self.soft_dof_limits = (mid - half, mid + half)

        link = lambda name: self.robot.get_link(name).idx_local  # noqa: E731
        self.feet_links_idx = [link(n) for n in env_cfg["feet_link_names"]]
        self.hands_links_idx = [link(n) for n in env_cfg["hand_link_names"]]
        self.base_link_idx = link(env_cfg["base_link_name"])
        self.robot_mass = float(self.robot.get_mass())
        self.link_masses = self.robot.get_links_mass().reshape(-1, self.robot.n_links)[0]  # 全身の重心の計算に使う

        # ------------------------------------------------------------------
        # バッファ
        # ------------------------------------------------------------------
        n, a = num_envs, self.num_actions
        f = dict(dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in joint_names], **f
        )
        self.init_dof_pos_all = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]], **f
        )
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], **f)

        self.base_pos = torch.zeros((n, 3), **f)
        self.base_quat = torch.zeros((n, 4), **f)
        self.base_quat[:, 0] = 1.0
        self.base_euler = torch.zeros((n, 3), **f)
        self.base_lin_vel = torch.zeros((n, 3), **f)
        self.base_ang_vel = torch.zeros((n, 3), **f)
        self.projected_gravity = torch.zeros((n, 3), **f)
        self.projected_gravity[:, 2] = -1.0
        self.dof_pos = torch.zeros((n, a), **f)
        self.dof_vel = torch.zeros((n, a), **f)
        self.last_dof_vel = torch.zeros((n, a), **f)
        self.actions = torch.zeros((n, a), **f)
        self.last_actions = torch.zeros((n, a), **f)
        self.contact_forces = torch.zeros((n, self.robot.n_links, 3), **f)
        self.feet_pos = torch.zeros((n, 2, 3), **f)
        self.hands_pos = torch.zeros((n, 2, 3), **f)
        self.rew_buf = torch.zeros((n,), **f)
        self.reset_buf = torch.ones((n,), dtype=gs.tc_bool, device=gs.device)
        self.time_out_buf = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)

        # タスクの状態
        self.spawn_x = torch.full((n,), -0.45, **f)
        self.progress = torch.zeros((n,), **f)  # 現在の到達度（0〜1）
        self.best_progress = torch.zeros((n,), **f)  # このエピソードでの最高到達度
        self.progress_gain = torch.zeros((n,), **f)  # 今のステップで最高到達度を更新した量
        self.feet_on_box = torch.zeros((n, 2), dtype=gs.tc_bool, device=gs.device)
        self.hands_on_box = torch.zeros((n, 2), dtype=gs.tc_bool, device=gs.device)
        self.on_top = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.succeeded = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)  # このエピソードで登れたか
        self.just_succeeded = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.fell = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.hands_touched = torch.zeros((n, 2), dtype=gs.tc_bool, device=gs.device)  # このエピソードで手が台に触れたか
        self.hands_first_touch = torch.zeros((n, 2), dtype=gs.tc_bool, device=gs.device)
        self.in_contact = torch.ones((n,), dtype=gs.tc_bool, device=gs.device)  # 体のどこかが床か台に触れているか
        self.feet_lift = torch.zeros((n, 2), **f)
        self.com_pos = torch.zeros((n, 3), **f)  # 全身の重心の位置
        self.base_vel_world = torch.zeros((n, 3), **f)  # 胴体の速度（ワールド座標）
        self.support_steps = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)  # 手か足が台の上に触れ続けているステップ数
        self.flight_steps = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)  # 体のどこも触れていない状態が続いたステップ数
        self.has_landed = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)  # 置き直した後、一度でも床か台に触れたか
        # 確認用：到達度の更新のうち、跳んでいる最中に起きた割合（移動平均）
        self._gain_total = torch.zeros((), **f)
        self._gain_jumping = torch.zeros((), **f)

        # 台の高さのカリキュラム（環境ごとの段階）
        cc = curriculum_cfg
        self.num_levels = int(round((cc["height_max"] - cc["height_min"]) / cc["height_step"])) + 1
        self.levels = torch.randint(0, cc["init_max_level"] + 1, (n,), device=gs.device)
        self.box_height = torch.zeros((n,), **f)
        self.fixed_height = None  # 評価時に台の高さを固定するためのもの
        # 開始姿勢の再利用：以前の方策が到達した「登る途中・登った後」の姿勢から始めるエピソードを混ぜる
        self.height_override = torch.full((n,), float("nan"), **f)  # その環境の台の高さを固定する（nan なら段階どおり）
        self.is_replay = torch.zeros((n,), dtype=gs.tc_bool, device=gs.device)
        self.replay_qpos = None
        replay_file = env_cfg["reset"].get("replay_file")
        if replay_file:
            if not os.path.isabs(replay_file):
                replay_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), replay_file)
            data = torch.load(replay_file, map_location=gs.device)
            self.replay_qpos = data["qpos"].to(**f)
            self.replay_height = data["height"].to(**f)
            # 姿勢の種類（0: 片足が台の上, 1: 台の上に立つ, 2: 手が台の上）
            self.replay_kind = data.get("kind", torch.zeros_like(data["height"])).to(device=gs.device)
        self._update_box_height(None)

        # 補助の力のカリキュラム（学習の経過に応じて減らす）
        self.common_step_counter = 0
        self.assist_scale = cc["assist_init"]
        self.success_rate_ema = 0.0

        # 観測ノイズ
        noise = obs_cfg["noise_scales"]
        level = obs_cfg["noise_level"] if obs_cfg["add_noise"] else 0.0
        self.noise_scale_vec = torch.cat(
            (
                torch.full((3,), noise["ang_vel"] * self.obs_scales["ang_vel"]),
                torch.full((3,), noise["gravity"]),
                torch.full((a,), noise["dof_pos"] * self.obs_scales["dof_pos"]),
                torch.full((a,), noise["dof_vel"] * self.obs_scales["dof_vel"]),
                torch.zeros(a),  # 前回の行動
                torch.full((4,), noise["box"]),  # 台との位置関係・向き
            )
        ).to(**f) * level

        # 報酬関数（dense は毎ステップの値に dt を掛ける。sparse はそのまま与える）
        self.reward_scales = {}
        self.reward_functions, self.episode_sums = {}, {}
        for name, scale in reward_cfg["dense_scales"].items():
            self.reward_scales[name] = scale * self.dt
        for name, scale in reward_cfg["sparse_scales"].items():
            self.reward_scales[name] = scale
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((n,), **f)

        self.extras = dict()
        self.reset()

    # ----------------------------------------------------------------------
    # カリキュラム
    # ----------------------------------------------------------------------
    def _update_box_height(self, envs_idx):
        cc = self.curriculum_cfg
        height = cc["height_min"] + self.levels.to(gs.tc_float) * cc["height_step"]
        if self.fixed_height is not None:
            height = torch.full_like(height, self.fixed_height)
        height = torch.where(torch.isnan(self.height_override), height, self.height_override)
        if envs_idx is None:
            self.box_height.copy_(height)
        else:
            torch.where(envs_idx, height, self.box_height, out=self.box_height)
        box_pos = torch.stack(
            (
                torch.full_like(self.box_height, self.box_size[0] / 2),
                torch.zeros_like(self.box_height),
                self.box_height - self.box_size[2] / 2,
            ),
            dim=1,
        )
        self.box.set_pos(box_pos)

    def set_box_height(self, height):
        """評価用：台の高さを固定する [m]。"""
        self.fixed_height = float(height)
        self._update_box_height(None)

    def _update_curriculum(self, envs_idx):
        """エピソードが終わった環境の段階を更新する（登れたら上げ、ほとんど進めなかったら下げる）。"""
        cc = self.curriculum_cfg
        up = envs_idx & self.succeeded
        down = envs_idx & ~self.succeeded & (self.best_progress < cc["level_down_progress"])
        self.levels += up.to(self.levels.dtype)
        self.levels -= down.to(self.levels.dtype)
        # 最上段を超えたら、上の方の段階からランダムに選び直す（低い台を忘れないように）
        over = self.levels >= self.num_levels
        random_levels = torch.randint(
            max(0, self.num_levels - cc["top_levels_resample"]), self.num_levels, (self.num_envs,), device=gs.device
        )
        self.levels = torch.where(over, random_levels, self.levels).clamp(min=0)

        # 成功率の移動平均（ログ用）
        n_done = envs_idx.sum()
        rate = torch.where(n_done > 0, (envs_idx & self.succeeded).sum() / n_done.clamp(min=1), self.success_rate_ema)
        self.success_rate_ema = 0.99 * float(self.success_rate_ema) + 0.01 * float(rate)

    def _update_assist(self):
        cc = self.curriculum_cfg
        frac = min(self.common_step_counter / max(cc["assist_anneal_steps"], 1), 1.0)
        self.assist_scale = cc["assist_init"] * (1.0 - frac)

    # ----------------------------------------------------------------------
    # 1 ステップ進める
    # ----------------------------------------------------------------------
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # 補助の力：手か足が台の上に触れていて、まだ台の上に立てていない環境だけ、腰を上向きに持ち上げる
        # （登り始めた体を下から支える「補助者」の役割。台から離れているときはかけない）
        self._update_assist()
        if self.assist_scale > 0.0:
            near = (self.feet_on_box.any(dim=1) | self.hands_on_box.any(dim=1)) & ~self.on_top
            force = torch.zeros((self.num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)
            force[:, 0, 2] = near.to(gs.tc_float) * self.assist_scale * self.robot_mass * 9.81
            self.robot.apply_links_external_wrench(force=force, links_idx_local=[self.base_link_idx])

        self.scene.step()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 状態の更新
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(self.base_quat, rpy=True, degrees=False)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_vel_world = self.robot.get_vel()
        self.base_lin_vel = transform_by_quat(self.base_vel_world, inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.feet_pos = self.robot.get_links_pos(self.feet_links_idx)
        self.hands_pos = self.robot.get_links_pos(self.hands_links_idx)
        links_com = self.robot.get_links_pos(ref=gs.link_ref_frame.link_COM)
        self.com_pos = (links_com * self.link_masses[None, :, None]).sum(dim=1) / self.link_masses.sum()
        self._update_task_state()

        # 終了判定
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf | self.fell | self.scene.rigid_solver.get_error_envs_mask()

        # 報酬
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.extras["time_outs"] = self.time_out_buf.to(dtype=gs.tc_float)
        # 評価用：リセットで消える前のタスクの状態を残しておく
        self.last_best_progress = self.best_progress.clone()
        self.last_just_succeeded = self.just_succeeded.clone()
        self.last_on_top = self.on_top.clone()
        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_task_state(self):
        h = self.box_height
        tc = self.env_cfg["task"]
        # 足・手が台の上面に乗っているか（台の上にあり、上面より上にあり、力がかかっている）
        feet_force = torch.norm(self.contact_forces[:, self.feet_links_idx], dim=-1)
        hands_force = torch.norm(self.contact_forces[:, self.hands_links_idx], dim=-1)
        self.feet_on_box = (
            (self.feet_pos[:, :, 0] > tc["on_box_margin_x"])
            & (self.feet_pos[:, :, 2] > h[:, None] - 0.01)
            & (feet_force > 5.0)
        )
        self.hands_on_box = (
            (self.hands_pos[:, :, 0] > -0.05)
            & (self.hands_pos[:, :, 2] > h[:, None] - 0.05)
            & (hands_force > 5.0)
        )

        # 到達度（0〜1）= 足を上げた高さ × 重み + 重心の上がった割合 × 重み + 台に乗った足の数の割合 × 重み
        #   重心の上がった割合：立っているときの重心の高さから、台の高さぶん上がれば 1。
        #   次の両方を満たすときだけ数える（跳び上がって一瞬だけ重心が上がった分を数えないため）：
        #     ・手か足が台の上に support_min_steps ステップ以上触れ続けている
        #     ・胴体の上下の速さが max_rise_speed 未満（台にかけた足で蹴って跳び上がる動きを除く）
        supported = self.feet_on_box.any(dim=1) | self.hands_on_box.any(dim=1)
        self.support_steps = torch.where(supported, self.support_steps + 1, torch.zeros_like(self.support_steps))
        slow = torch.abs(self.base_vel_world[:, 2]) < tc.get("max_rise_speed", 0.8)
        steady = ((self.support_steps >= tc.get("support_min_steps", 5)) & slow).to(gs.tc_float)
        p_com = ((self.com_pos[:, 2] - tc.get("stand_com_height", 0.68)) / h.clamp(min=0.05)).clamp(0.0, 1.0) * steady
        p_feet = self.feet_on_box.to(gs.tc_float).mean(dim=1)
        # 足を上げた高さ：台の近くにある足を、台の高さまで上げた割合（最初の手がかり。重みは小さくする）
        self.feet_lift = ((self.feet_pos[:, :, 2] - tc["foot_rest_height"]) / h[:, None].clamp(min=0.05)).clamp(0.0, 1.0)
        near_box = (self.feet_pos[:, :, 0] > -tc.get("foot_near_box_x", 0.3)).to(gs.tc_float)
        p_lift = (self.feet_lift * near_box).max(dim=1).values
        # 重心が台の上へ前進した割合（v7）：重心の水平位置が com_fwd_start から com_fwd_target まで進めば 1。
        #   重心の高さと同じく、台に触れ続けていて跳んでいないときだけ数える（台に乗るには、まず重心を台の上の足へ運ぶ必要がある）
        p_fwd = (
            (self.com_pos[:, 0] - tc.get("com_fwd_start", -0.3))
            / (tc.get("com_fwd_target", 0.2) - tc.get("com_fwd_start", -0.3))
        ).clamp(0.0, 1.0) * steady
        w = tc["progress_weights"]
        if len(w) == 4:
            self.progress = w[0] * p_lift + w[1] * p_com + w[2] * p_feet + w[3] * p_fwd
        elif len(w) == 3:
            self.progress = w[0] * p_lift + w[1] * p_com + w[2] * p_feet
        else:
            self.progress = w[0] * p_com + w[1] * p_feet
        # 途中の姿勢から始めたエピソードは、始めた時点の到達度を基準にする（何もせずに報酬が出ないように）
        fresh_replay = self.is_replay & (self.episode_length_buf <= 1)
        self.best_progress = torch.where(fresh_replay, self.progress, self.best_progress)
        self.progress_gain = (self.progress - self.best_progress).clamp(min=0.0)
        self.best_progress = torch.maximum(self.best_progress, self.progress)

        # 台の上に立てているか：両足が台の上、腰が十分高く、体が起きている
        upright = torch.norm(self.projected_gravity[:, :2], dim=1) < tc["upright_tilt"]
        high = self.base_pos[:, 2] > h + tc["stand_height"] * 0.8
        self.on_top = self.feet_on_box.all(dim=1) & high & upright
        self.succeeded |= fresh_replay & self.on_top  # 台の上から始めた場合は、成功の報酬を出さない
        self.just_succeeded = self.on_top & ~self.succeeded
        self.succeeded |= self.on_top

        # 手が初めて台の上に触れたか（腕を使うことを促す一度きりの報酬に使う）
        self.hands_touched |= fresh_replay[:, None] & self.hands_on_box
        self.hands_first_touch = self.hands_on_box & ~self.hands_touched
        self.hands_touched |= self.hands_on_box

        # 体のどこかが床か台に触れているか（触れていない＝跳んでいる）
        self.in_contact = (torch.norm(self.contact_forces, dim=-1) > 1.0).any(dim=1)
        self.flight_steps = torch.where(self.in_contact, torch.zeros_like(self.flight_steps), self.flight_steps + 1)
        self.has_landed |= self.in_contact

        # 確認用：到達度の更新が「跳んでいる最中（どこも触れていない、または上向きに 1 m/s 以上で上昇中）」に起きた割合
        jumping = (~self.in_contact) | (self.base_lin_vel[:, 2] > 1.0)
        self._gain_total = 0.999 * self._gain_total + self.progress_gain.sum()
        self._gain_jumping = 0.999 * self._gain_jumping + (self.progress_gain * jumping.to(gs.tc_float)).sum()

        # 転倒：台の手前で腰が低すぎる／大きく横に倒れた／前に倒れすぎた（台に覆いかぶさる）／後ろに倒れた
        over_box = self.base_pos[:, 0] > 0.0
        support = torch.where(over_box, h, torch.zeros_like(h))
        too_low = (self.base_pos[:, 2] - support) < torch.where(
            over_box, torch.full_like(h, tc["min_height_over_box"]), torch.full_like(h, tc["min_height_on_ground"])
        )
        # 一度台の上に立った後に、台から降りてしまった（両足が台から離れ、腰が台の上面近くまで下がった）
        left_box = self.succeeded & ~self.feet_on_box.any(dim=1) & (self.base_pos[:, 2] < h + tc.get("left_box_height", 0.4))
        # 跳躍：体のどこも床や台に触れていない状態が max_flight_steps ステップ続いた
        # （置き直した直後、最初に着地するまでは数えない）
        max_flight = tc.get("max_flight_steps")
        jumped = torch.zeros_like(too_low)
        if max_flight is not None:
            jumped = (self.flight_steps >= max_flight) & self.has_landed
        self.fell = (
            too_low
            | left_box
            | jumped
            | (torch.abs(self.base_euler[:, 0]) > tc["max_roll"])
            | (self.base_euler[:, 1] > tc["max_pitch_forward"])
            | (self.base_euler[:, 1] < -tc["max_pitch_back"])
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf, "critic": self.privileged_obs_buf}, batch_size=[self.num_envs])

    def _update_observation(self):
        yaw = self.base_euler[:, 2]
        box_obs = torch.stack(
            (
                (0.0 - self.base_pos[:, 0]),  # 台の正面までの距離 [m]
                (self.box_height - self.base_pos[:, 2]),  # 台の上面と腰の高さの差 [m]
                torch.sin(yaw),
                torch.cos(yaw),
            ),
            dim=1,
        )
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 23
                self.dof_vel * self.obs_scales["dof_vel"],  # 23
                self.actions,  # 23
                box_obs,  # 4
            ),
            dim=-1,
        )
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # 価値関数にだけ与える情報：胴体の速度・高さ・手足の接地・到達度・補助の強さ
        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_pos[:, 2:3],  # 1
                self.feet_on_box.to(gs.tc_float),  # 2
                self.hands_on_box.to(gs.tc_float),  # 2
                self.best_progress[:, None],  # 1
                torch.full((self.num_envs, 1), self.assist_scale, dtype=gs.tc_float, device=gs.device),  # 1
            ),
            dim=-1,
        )

    # ----------------------------------------------------------------------
    # リセット
    # ----------------------------------------------------------------------
    def _reset_idx(self, envs_idx=None):
        all_envs = envs_idx is None
        mask = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device) if all_envs else envs_idx

        # 学習ログ（報酬の内訳とカリキュラムの状態）
        n_done = mask.sum()
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            mean = torch.where(n_done > 0, value[mask].sum() / n_done.clamp(min=1), 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(mask, 0.0)

        # カリキュラムの更新（評価時は台の高さを固定する）
        if not all_envs and self.fixed_height is None:
            self._update_curriculum(mask & ~self.is_replay)  # 途中の姿勢から始めたエピソードは段階の更新に使わない

        # 一部の環境は、保存しておいた「登る途中・登った後」の姿勢から始める
        rc = self.env_cfg["reset"]
        replay = torch.zeros_like(mask)
        replay_idx = None
        if self.replay_qpos is not None and self.fixed_height is None:
            replay = mask & (torch.rand((self.num_envs,), device=gs.device) < rc.get("replay_prob", 0.0))
            replay_idx = torch.randint(0, self.replay_qpos.shape[0], (self.num_envs,), device=gs.device)
            override = torch.where(replay, self.replay_height[replay_idx], torch.full_like(self.box_height, float("nan")))
        else:
            override = torch.full_like(self.box_height, float("nan"))
        self.height_override = torch.where(mask, override, self.height_override)
        self.is_replay = torch.where(mask, replay, self.is_replay)
        self._update_box_height(None if all_envs else mask)
        heights = self.curriculum_cfg["height_min"] + self.levels.to(gs.tc_float) * self.curriculum_cfg["height_step"]
        self.extras["episode"]["Curriculum/mean_height"] = heights.mean()
        self.extras["episode"]["Curriculum/max_height"] = heights.max()
        self.extras["episode"]["Curriculum/assist"] = torch.tensor(self.assist_scale)
        self.extras["episode"]["Curriculum/success_rate"] = torch.tensor(self.success_rate_ema)
        self.extras["episode"]["Check/progress_gain_while_jumping"] = self._gain_jumping / self._gain_total.clamp(min=1e-6)

        # ロボットを台の手前に置き直す（距離と向きを少しランダムにする）
        spawn_x = -gs_rand(*rc["distance_range"], (self.num_envs,))
        spawn_y = gs_rand(-rc["lateral_range"], rc["lateral_range"], (self.num_envs,))
        spawn_yaw = gs_rand(-rc["yaw_range"], rc["yaw_range"], (self.num_envs,))
        torch.where(mask, spawn_x, self.spawn_x, out=self.spawn_x)
        quat = xyz_to_quat(
            torch.stack((torch.zeros_like(spawn_yaw), torch.zeros_like(spawn_yaw), spawn_yaw), dim=1), rpy=True
        )
        qpos = torch.cat(
            (
                torch.stack((spawn_x, spawn_y, torch.full_like(spawn_x, rc["height"])), dim=1),
                quat,
                self.init_dof_pos_all.expand(self.num_envs, -1),
            ),
            dim=1,
        )
        if replay_idx is not None:
            qpos = torch.where(replay[:, None], self.replay_qpos[replay_idx], qpos)
        qpos = torch.where(mask[:, None], qpos, self.robot.get_qpos())
        self.robot.set_qpos(qpos, zero_velocity=False, skip_forward=True)
        if all_envs:
            self.robot.zero_all_dofs_velocity()
        else:
            vel = self.robot.get_dofs_velocity()
            self.robot.set_dofs_velocity(torch.where(mask[:, None], torch.zeros_like(vel), vel), skip_forward=True)

        # バッファのリセット
        m1 = mask[:, None]
        for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel, self.actions, self.last_actions,
                    self.last_dof_vel):
            buf.masked_fill_(m1, 0.0)
        torch.where(m1, qpos[:, self.motors_dof_idx + 1], self.dof_pos, out=self.dof_pos)  # qpos は姿勢が 4 要素なので +1
        self.base_pos = torch.where(m1, qpos[:, :3], self.base_pos)
        self.base_quat = torch.where(m1, qpos[:, 3:7], self.base_quat)
        self.base_euler = torch.where(m1, quat_to_xyz(qpos[:, 3:7], rpy=True, degrees=False), self.base_euler)
        self.projected_gravity = torch.where(
            m1, transform_by_quat(self.global_gravity.expand(self.num_envs, -1), inv_quat(qpos[:, 3:7])),
            self.projected_gravity,
        )
        self.episode_length_buf.masked_fill_(mask, 0)
        for buf in (self.progress, self.best_progress, self.progress_gain):
            buf.masked_fill_(mask, 0.0)
        for buf in (self.succeeded, self.on_top, self.just_succeeded, self.fell):
            buf.masked_fill_(mask, False)
        if replay_idx is not None:
            # 台の上に立った姿勢から始めた場合は、最初から「登れた」扱いにする（成功の報酬は出さず、台から降りたら失敗）
            self.succeeded |= replay & (self.replay_kind[replay_idx] == 1)
        self.hands_touched.masked_fill_(m1, False)
        self.hands_first_touch.masked_fill_(m1, False)
        self.in_contact.masked_fill_(mask, True)
        self.support_steps.masked_fill_(mask, 0)
        self.flight_steps.masked_fill_(mask, 0)
        self.has_landed.masked_fill_(mask, False)
        self.feet_on_box.masked_fill_(m1, False)
        self.hands_on_box.masked_fill_(m1, False)
        self.reset_buf.masked_fill_(mask, True)

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ----------------------------------------------------------------------
    # 報酬関数
    # ----------------------------------------------------------------------
    # --- タスク（sparse：dt を掛けない） ---
    def _reward_progress(self):
        # 最高到達度を更新したぶんだけ報酬（エピソード全体で最大 1 × 係数）
        return self.progress_gain

    def _reward_success(self):
        # 初めて台の上に立てたとき
        return self.just_succeeded.to(gs.tc_float)

    def _reward_termination(self):
        # 転倒して終わったとき（時間切れは含まない）。このエピソードで得た到達度の報酬も取り消す
        # （足を台に蹴り上げてすぐ転ぶ、のように「一瞬だけ到達度を稼いで転ぶ」動きで得をしないようにする）
        fell = (self.fell & ~self.time_out_buf).to(gs.tc_float)
        refund = self.best_progress * self.reward_cfg["sparse_scales"]["progress"] / abs(self.reward_scales["termination"])
        return fell * (1.0 + refund)

    def _reward_hands_touch(self):
        # 手が初めて台の上に触れたとき（片手ごとに一度だけ。台が低いときは与えない）
        tall = (self.box_height >= self.reward_cfg["hands_touch_min_height"]).to(gs.tc_float)
        return self.hands_first_touch.to(gs.tc_float).sum(dim=1) * tall

    # --- タスク（dense：毎ステップ） ---
    def _reward_on_top(self):
        # 台の上に立ち続けている間
        return self.on_top.to(gs.tc_float)

    def _reward_hands_on_box(self):
        # 台の上に立つ前に、手を台の上についている（腕を使うことを促す）
        return self.hands_on_box.to(gs.tc_float).sum(dim=1) * (~self.on_top).to(gs.tc_float)

    def _reward_trailing_foot(self):
        # 片足だけが台の上にあるとき、もう一方の足を台の高さまで上げるほど報酬（後ろ足を台に引き上げさせる）
        one_foot = (self.feet_on_box.sum(dim=1) == 1).to(gs.tc_float)
        other_lift = (self.feet_lift * (~self.feet_on_box).to(gs.tc_float)).sum(dim=1)
        return one_foot * other_lift

    def _reward_heading(self):
        # 台の方向（+x）から大きく向きを変える（体をひねって回り込む・回転しながら跳ぶ）のを抑える
        yaw = self.base_euler[:, 2]
        return torch.square((torch.abs(yaw) - self.reward_cfg.get("heading_free", 0.5)).clamp(min=0.0))

    def _reward_lean_back(self):
        # 後ろに大きくのけぞる姿勢（片足を台にかけたまま体を後ろに倒して踏ん張る）を抑える
        return torch.square((-self.base_euler[:, 1] - 0.2).clamp(min=0.0))

    def _reward_alive(self):
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    # --- 動きの質（dense） ---
    def _reward_flight(self):
        # 体のどこも床や台に触れていない（跳んでいる）
        return (~self.in_contact).to(gs.tc_float)

    def _reward_lin_vel_z(self):
        # 胴体の速すぎる上下動（跳び上がる・飛び降りる）を抑える
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_pos_limits(self):
        lower, upper = self.soft_dof_limits
        out_of_limits = -(self.dof_pos - lower).clip(max=0.0) + (self.dof_pos - upper).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_contact_force(self):
        # 強すぎる衝撃（飛びつく・叩きつける動き）を抑える
        excess = torch.norm(self.contact_forces, dim=-1) - self.reward_cfg["max_contact_force"]
        return torch.sum(excess.clip(min=0.0), dim=1)

    def _reward_upright_on_top(self):
        # 台の上では体を起こして立つ
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.on_top.to(gs.tc_float)
