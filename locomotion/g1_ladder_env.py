"""ヒューマノイド Unitree G1（23 関節）が、傾いたはしごを手と足を使って登る強化学習の環境。

台へのよじ登り（g1_climb_env.py）との主な違い:
  - はしご：横棒（段）を等間隔に並べたもの。傾き（床からの角度）は環境ごとに変えられ、60°（急な階段くらい）から
    90°（垂直）までのカリキュラムにする。垂直に近いほど、手で体を引きつけないと後ろに倒れるので、手が必須になる
  - 握る動作の簡略化：手のひらが横棒のすぐ近くにあるときに「握る」指令を出すと、手のひらをその位置に
    ばね（上限つきの力）で引きつける。G1（23 関節版）の手には指がなく横棒を握れないため、握る力を
    ばねで表す。力に上限（既定 250 N）があるので、握力の限界も表せる。指令をやめると離す
  - 行動：23 関節の目標角度に加えて、左右の手の「握る」指令（正なら握る）の 2 つ。合計 25

座標：はしごの根元（床の位置）が原点。はしごは +x の方向に傾いて立ち、横棒は y 方向に伸びる。
ロボットは x < 0 から +x 方向（はしごの方）を向いて始める。
横棒 k の中心 = s_k × (cos θ, 0, sin θ)、s_k = rung_first + k × rung_spacing（はしごに沿った距離）。
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


class G1LadderEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, curriculum_cfg, show_viewer=False, camera_res=None,
                 rail_angle_deg=None):
        self.num_envs: int = num_envs
        self.num_joints = len(env_cfg["joint_names"])
        self.num_actions = self.num_joints + 2  # 関節 23 + 左右の手の「握る」指令 2
        self.device = gs.device
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.cfg = env_cfg  # rsl_rl の OnPolicyRunner が参照する
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.curriculum_cfg = curriculum_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        lc = env_cfg["ladder"]
        self.num_rungs = lc["num_rungs"]
        self.rung_s = torch.tensor(
            [lc["rung_first"] + k * lc["rung_spacing"] for k in range(self.num_rungs)], dtype=gs.tc_float, device=gs.device
        )

        # ------------------------------------------------------------------
        # シーン：床・はしごの横棒・ロボット
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
                camera_pos=(-1.8, -2.8, 2.5),  # v11a：高い段まで映るように引いて、見る点を上げた
                camera_lookat=(0.0, 0.0, 1.7),
                camera_fov=50,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        # 横棒は 1 本ずつ別の物体にする（Genesis では、固定した物体を環境ごとに傾けられないため。
        # 横棒は y 方向に伸びる棒なので、はしごの傾きが変わっても向きは同じで、位置だけを変えればよい）
        t = lc["rung_thickness"]
        self.rungs = [
            self.scene.add_entity(
                gs.morphs.Box(pos=(0.0, 0.0, float(self.rung_s[k])), size=(t, lc["rung_width"], t), fixed=True),
                surface=gs.surfaces.Default(color=(0.75, 0.6, 0.35)),
            )
            for k in range(self.num_rungs)
        ]
        # 見た目だけの支柱（評価で、はしごの傾きが決まっているときだけ置く。衝突はしない）
        if rail_angle_deg is not None:
            th = math.radians(rail_angle_deg)
            length = float(self.rung_s[-1]) + 0.2
            for side in (-1, 1):
                self.scene.add_entity(
                    gs.morphs.Box(
                        pos=(length / 2 * math.cos(th), side * (lc["rung_width"] / 2 + 0.02), length / 2 * math.sin(th)),
                        size=(0.04, 0.04, length),
                        euler=(0.0, 90.0 - rail_angle_deg, 0.0),
                        fixed=True,
                        collision=False,
                    ),
                    surface=gs.surfaces.Default(color=(0.6, 0.45, 0.25)),
                )
        # ロボットのモデル（urdf_path がファイル名だけなら assets/unitree_g1 の中を探す）
        urdf = env_cfg.get("urdf_path") or DEFAULT_URDF
        if not os.path.isabs(urdf):
            urdf = os.path.join(os.path.dirname(DEFAULT_URDF), urdf)
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=urdf, pos=(-0.5, 0.0, 0.8)))

        self.camera = None
        if camera_res is not None:
            self.camera = self.scene.add_camera(
                res=camera_res, pos=(-1.8, -2.8, 2.5), lookat=(0.0, 0.0, 1.7), fov=50, GUI=False, env_idx=0
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
        lower, upper = lower.reshape(-1, self.num_joints)[0], upper.reshape(-1, self.num_joints)[0]
        mid, half = (lower + upper) / 2, (upper - lower) / 2 * reward_cfg["soft_dof_pos_limit"]
        self.soft_dof_limits = (mid - half, mid + half)

        link = lambda name: self.robot.get_link(name).idx_local  # noqa: E731
        self.feet_links_idx = [link(n) for n in env_cfg["feet_link_names"]]
        self.hands_links_idx = [link(n) for n in env_cfg["hand_link_names"]]
        self.base_link_idx = link(env_cfg["base_link_name"])
        # 手足以外のリンク（胴体・腰・脚・腕。はしごにもたれかかる動きを抑えるのに使う）
        limb_names = set(env_cfg["feet_link_names"]) | set(env_cfg["hand_link_names"]) | set(env_cfg.get("ankle_link_names", []))
        self.body_links_idx = [l.idx_local for l in self.robot.links if l.name not in limb_names]
        self.robot_mass = float(self.robot.get_mass())
        self.link_masses = self.robot.get_links_mass().reshape(-1, self.robot.n_links)[0]

        # ------------------------------------------------------------------
        # バッファ
        # ------------------------------------------------------------------
        n, a, j = num_envs, self.num_actions, self.num_joints
        f = dict(dtype=gs.tc_float, device=gs.device)
        b = dict(dtype=gs.tc_bool, device=gs.device)
        self.default_dof_pos = torch.tensor([env_cfg["default_joint_angles"][name] for name in joint_names], **f)
        self.init_dof_pos_all = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]], **f
        )
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], **f)
        self.palm_offset = torch.tensor(env_cfg["palm_offset"], **f)  # 手のリンクの原点（手首）から手のひらの中心まで

        self.base_pos = torch.zeros((n, 3), **f)
        self.base_quat = torch.zeros((n, 4), **f)
        self.base_quat[:, 0] = 1.0
        self.base_euler = torch.zeros((n, 3), **f)
        self.base_lin_vel = torch.zeros((n, 3), **f)
        self.base_ang_vel = torch.zeros((n, 3), **f)
        self.base_vel_world = torch.zeros((n, 3), **f)
        self.projected_gravity = torch.zeros((n, 3), **f)
        self.projected_gravity[:, 2] = -1.0
        self.dof_pos = torch.zeros((n, j), **f)
        self.dof_vel = torch.zeros((n, j), **f)
        self.last_dof_vel = torch.zeros((n, j), **f)
        self.actions = torch.zeros((n, a), **f)
        self.last_actions = torch.zeros((n, a), **f)
        self.contact_forces = torch.zeros((n, self.robot.n_links, 3), **f)
        self.feet_pos = torch.zeros((n, 2, 3), **f)
        self.palms_pos = torch.zeros((n, 2, 3), **f)
        self.com_pos = torch.zeros((n, 3), **f)
        self.rew_buf = torch.zeros((n,), **f)
        self.reset_buf = torch.ones((n,), **b)
        self.time_out_buf = torch.zeros((n,), **b)
        self.stalled = torch.zeros((n,), **b)  # 到達度が長い間更新されず、打ち切った（v5）
        self.violated = torch.zeros((n,), **b)  # 手と足の順番を守らず、打ち切った（v15e）
        self.last_violated = torch.zeros((n,), **b)
        self.stall_steps = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)  # 到達度の最高記録が更新されていないステップ数
        self.feet_best_rung = torch.full((n, 2), -1, dtype=gs.tc_int, device=gs.device)  # このエピソードで各足が乗った最高の段
        self.feet_hold_steps = torch.zeros((n, 2), dtype=gs.tc_int, device=gs.device)  # 各足が同じ段に乗り続けているステップ数（v6）
        self.feet_prev_rung = torch.full((n, 2), -1, dtype=gs.tc_int, device=gs.device)
        self.hands_best_rung = torch.full((n, 2), -1, dtype=gs.tc_int, device=gs.device)  # このエピソードで各手が握った最高の段（v7）
        self.new_hand_rungs = torch.zeros((n,), **f)  # 今のステップで、手が初めて握った（1 つ上の）段の数（v7）
        self.new_rungs = torch.zeros((n,), **f)  # 今のステップで、足が初めて乗った段の数（v5）
        # 手と足の順番（v15）
        self.hand_turn = torch.zeros((n,), **b)  # 今が「手の番」か（手と足の段の差が小さい）
        self.out_of_turn = torch.zeros((n,), **f)  # 今のステップで、手の番なのに足の最高の段を上げた数
        self.feet_turn_best = torch.full((n,), -1, dtype=gs.tc_int, device=gs.device)  # 足の番に上げた足の最高の段
        self.feet_seen_best = torch.full((n, 2), -1, dtype=gs.tc_int, device=gs.device)  # 各足が 0.2 秒以上乗った最高の段（腰との高さは問わない。v15d）
        self.release_block = torch.zeros((n, 2), dtype=gs.tc_int, device=gs.device)  # 強制的に離した手が握れない残りステップ数（v15b）
        self.forced_releases = torch.zeros((n,), **f)  # 今のステップで強制的に離した手の数（v15b。記録用）
        self.body_best = torch.zeros((n,), **f)  # 到達度のうち重心・足の分の最高記録（手の番の間は増やさない）
        self.reach_next_best = torch.zeros((n,), **f)  # 手の番の間、手が次の段に近づいた度合いの最高記録
        self.reach_next_rung = torch.full((n,), -1, dtype=gs.tc_int, device=gs.device)  # その「次の段」
        self.reach_next_gain = torch.zeros((n,), **f)
        self.episode_length_buf = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)

        # はしごの状態（環境ごと）
        self.ladder_angle = torch.full((n,), math.radians(75.0), **f)  # 床からの角度 [rad]
        self.ladder_u = torch.zeros((n, 3), **f)  # はしごに沿った上向きの単位ベクトル
        self.ladder_nrm = torch.zeros((n, 3), **f)  # はしごの面に垂直で、ロボットの側を向く単位ベクトル
        self.target_height = torch.ones((n,), **f)  # 目標の段の高さ [m]
        # 目標の段のカリキュラム（v12）：環境ごとの目標の段。床からの開始で成功したら 1 段上げ、ほとんど進めなければ 1 段下げる
        k_max = env_cfg["task"]["target_rung"]
        self.target_rung_env = torch.full((n,), curriculum_cfg.get("target_init", k_max), dtype=torch.int64, device=gs.device)
        self.episode_target = torch.full((n,), k_max, dtype=torch.int64, device=gs.device)  # 今のエピソードの目標の段

        # 握る
        self.grip = torch.zeros((n, 2), **b)  # 握っているか（左右）
        self.grip_anchor = torch.zeros((n, 2, 3), **f)  # 握った位置（手のひらを引きつける先）
        self.grip_rung = torch.zeros((n, 2), dtype=gs.tc_int, device=gs.device)  # 握っている段
        self.grip_force = torch.zeros((n, 2, 3), **f)
        self.palm_rung_dist = torch.zeros((n, 2), **f)  # 手のひらから一番近い横棒までの距離 [m]
        self.palm_rung_idx = torch.zeros((n, 2), dtype=gs.tc_int, device=gs.device)

        # タスクの状態
        self.feet_on_rung = torch.zeros((n, 2), **b)
        self.feet_rung_idx = torch.full((n, 2), -1, dtype=gs.tc_int, device=gs.device)  # 乗っている段（乗っていなければ -1）
        self.progress = torch.zeros((n,), **f)
        self.best_progress = torch.zeros((n,), **f)
        self.progress_gain = torch.zeros((n,), **f)
        self.on_target = torch.zeros((n,), **b)  # 目標の段に両足で立ち、手で握っている
        self.under_ladder = torch.zeros((n,), **b)  # はしごの下にもぐり込んだ
        self.succeeded = torch.zeros((n,), **b)
        self.just_succeeded = torch.zeros((n,), **b)
        self.fell = torch.zeros((n,), **b)
        self.first_grip = torch.zeros((n, 2), **b)  # 今のステップで、このエピソード初めて（高い段を）握った
        self.gripped_once = torch.zeros((n, 2), **b)
        self.in_contact = torch.ones((n,), **b)
        self.support_steps = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        self.flight_steps = torch.zeros((n,), dtype=gs.tc_int, device=gs.device)
        self.has_landed = torch.zeros((n,), **b)

        # はしごの傾きのカリキュラム（環境ごとの段階）
        cc = curriculum_cfg
        self.num_levels = int(round((cc["angle_max"] - cc["angle_min"]) / cc["angle_step"])) + 1
        self.levels = torch.randint(0, cc["init_max_level"] + 1, (n,), device=gs.device)
        self.fixed_angle = None  # 評価時に傾きを固定する [deg]
        self.eval_start = None  # 評価時の開始の仕方（None/"ground"：床から、"hang"：ぶら下がり、"stand"：床に立って握る）
        self.eval_hang_rung = None  # 評価時に "hang" で握る段（None ならランダム）
        self.angle_override = torch.full((n,), float("nan"), **f)
        self.is_review = torch.zeros((n,), **b)
        self.is_hang = torch.zeros((n,), **b)  # はしごにぶら下がった姿勢から始めたエピソードか（v3）
        self.is_hang_only = torch.zeros((n,), **b)  # そのうち、床に立って握った姿勢ではなく、ぶら下がった姿勢から始めたか（v15c）
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
                torch.full((j,), noise["dof_pos"] * self.obs_scales["dof_pos"]),
                torch.full((j,), noise["dof_vel"] * self.obs_scales["dof_vel"]),
                torch.zeros(a),  # 前回の行動
                torch.full((self._num_ladder_obs,), noise["ladder"]),
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

        # ぶら下がった姿勢から始めるための準備（v3）：腕を上げた姿勢での、腰から見た手のひらの位置を測っておく
        def arm_pose_info(arm_pose):
            pose = dict(env_cfg["default_joint_angles"])
            pose.update(arm_pose)
            dof_pos_all = torch.tensor([pose[joint.name] for joint in self.robot.joints[1:]], **f)
            qpos = torch.cat((torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], **f), dof_pos_all))
            self.robot.set_qpos(qpos.expand(n, -1))
            return dof_pos_all, self._palms()[0] - torch.tensor([0.0, 0.0, 1.0], **f)  # 関節角度, 腰から見た手のひらの位置 [2, 3]

        self.hang_cfg = env_cfg["reset"].get("hang")
        if self.hang_cfg:
            self.hang_dof_pos_all, self.hang_palm_offset = arm_pose_info(self.hang_cfg["arm_pose"])
        # 床に立ったまま、肩の高さの段を両手で握った姿勢から始めるための準備（v4）
        self.stand_cfg = env_cfg["reset"].get("stand_grip")
        if self.stand_cfg:
            self.stand_dof_pos_all, self.stand_palm_offset = arm_pose_info(self.stand_cfg["arm_pose"])

        self.extras = dict()
        self.reset()

    _num_ladder_obs = 18

    # ----------------------------------------------------------------------
    # はしご・カリキュラム
    # ----------------------------------------------------------------------
    def _update_ladder(self, envs_idx):
        """はしごの傾きを（envs_idx の環境だけ）決め直し、横棒を並べ直す。"""
        cc = self.curriculum_cfg
        angle_deg = cc["angle_min"] + self.levels.to(gs.tc_float) * cc["angle_step"]
        if self.fixed_angle is not None:
            angle_deg = torch.full_like(angle_deg, self.fixed_angle)
        angle_deg = torch.where(torch.isnan(self.angle_override), angle_deg, self.angle_override)
        angle = torch.deg2rad(angle_deg)
        if envs_idx is None:
            self.ladder_angle.copy_(angle)
        else:
            torch.where(envs_idx, angle, self.ladder_angle, out=self.ladder_angle)
        c, s = torch.cos(self.ladder_angle), torch.sin(self.ladder_angle)
        zero = torch.zeros_like(c)
        self.ladder_u = torch.stack((c, zero, s), dim=1)
        self.ladder_nrm = torch.stack((-s, zero, c), dim=1)
        self.target_height = self.rung_s[self.episode_target] * s
        for k, rung in enumerate(self.rungs):
            rung.set_pos(self.rung_s[k] * self.ladder_u)

    def set_eval_start(self, start, hang_rung=None):
        """評価用：開始の仕方を固定する（"ground"・"hang"・"stand"）。"""
        if start == "hang" and not self.hang_cfg:
            raise ValueError("この実験の設定には、ぶら下がった姿勢からの開始（reset.hang）がありません（v3 以降の版で使えます）。")
        if start == "stand" and not self.stand_cfg:
            raise ValueError("この実験の設定には、床に立って握った姿勢からの開始（reset.stand_grip）がありません（v4 以降の版で使えます）。")
        self.eval_start = start
        self.eval_hang_rung = hang_rung

    def set_ladder_angle(self, angle_deg):
        """評価用：はしごの傾きを固定する [deg]。"""
        self.fixed_angle = float(angle_deg)
        self._update_ladder(None)

    def _update_curriculum(self, envs_idx):
        cc = self.curriculum_cfg
        up = envs_idx & self.succeeded
        down = envs_idx & ~self.succeeded & (self.best_progress < cc["level_down_progress"])
        self.levels += up.to(self.levels.dtype)
        self.levels -= down.to(self.levels.dtype)
        over = self.levels >= self.num_levels
        random_levels = torch.randint(
            max(0, self.num_levels - cc["top_levels_resample"]), self.num_levels, (self.num_envs,), device=gs.device
        )
        self.levels = torch.where(over, random_levels, self.levels).clamp(min=0)
        # 目標の段（v12）：成功したら 1 段上げ（上限は task.target_rung）、目標の段にほど遠ければ 1 段下げる
        if "target_init" in cc:
            k_max = self.env_cfg["task"]["target_rung"]
            t_up = up & (self.episode_target == self.target_rung_env)
            t_down = envs_idx & ~self.succeeded & (self.best_progress < cc.get("target_down_progress", 0.5))
            self.target_rung_env = torch.where(t_up, (self.target_rung_env + 1).clamp(max=k_max), self.target_rung_env)
            self.target_rung_env = torch.where(t_down, (self.target_rung_env - 1).clamp(min=cc["target_init"]), self.target_rung_env)
        n_done = envs_idx.sum()
        rate = torch.where(n_done > 0, (envs_idx & self.succeeded).sum() / n_done.clamp(min=1), self.success_rate_ema)
        self.success_rate_ema = 0.99 * float(self.success_rate_ema) + 0.01 * float(rate)

    def _update_assist(self):
        cc = self.curriculum_cfg
        frac = min(self.common_step_counter / max(cc["assist_anneal_steps"], 1), 1.0)
        self.assist_scale = cc["assist_init"] * (1.0 - frac)

    def _ladder_coords(self, p):
        """位置 p [n, m, 3] を、はしごに沿った距離 s・面からの距離 nn・一番近い横棒の番号と距離に変換する。"""
        s = (p * self.ladder_u[:, None, :]).sum(dim=-1)
        nn = (p * self.ladder_nrm[:, None, :]).sum(dim=-1)
        ds = s[..., None] - self.rung_s  # [n, m, num_rungs]
        dist = torch.sqrt(ds**2 + nn[..., None] ** 2)
        dmin, kmin = dist.min(dim=-1)
        ds_near = torch.gather(ds, -1, kmin[..., None])[..., 0]
        return s, nn, kmin, dmin, ds_near

    # ----------------------------------------------------------------------
    # 1 ステップ進める
    # ----------------------------------------------------------------------
    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions[:, : self.num_joints] * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # 握る：手のひらを、握った位置にばねで引きつける（力の上限 = 握力の限界）
        gc = self.env_cfg["grip"]
        palms = self._palms()
        palm_vel = self.robot.get_links_vel(self.hands_links_idx)
        force = gc["stiffness"] * (self.grip_anchor - palms) - gc["damping"] * palm_vel
        norm = torch.norm(force, dim=-1, keepdim=True)
        force = force * torch.clamp(gc["max_force"] / norm.clamp(min=1e-6), max=1.0)
        self.grip_force = force * self.grip[..., None].to(gs.tc_float)
        wrench = self.grip_force

        # 補助の力：はしごに手か足が掛かっている間だけ、腰を上向きに支える（学習の初期だけ）
        self._update_assist()
        if self.assist_scale > 0.0:
            near = (self.grip.any(dim=1) | self.feet_on_rung.any(dim=1)) & ~self.on_target
            assist = torch.zeros((self.num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)
            assist[:, 0, 2] = near.to(gs.tc_float) * self.assist_scale * self.robot_mass * 9.81
            self.robot.apply_links_external_wrench(force=assist, links_idx_local=[self.base_link_idx])
        self.robot.apply_links_external_wrench(force=wrench, links_idx_local=self.hands_links_idx)

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
        self.palms_pos = self._palms()
        links_com = self.robot.get_links_pos(ref=gs.link_ref_frame.link_COM)
        self.com_pos = (links_com * self.link_masses[None, :, None]).sum(dim=1) / self.link_masses.sum()
        self._update_grip()
        self._update_task_state()

        # 終了判定
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = (
            self.time_out_buf | self.fell | self.stalled | self.violated | self.scene.rigid_solver.get_error_envs_mask()
        )

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
        self.last_on_target = self.on_target.clone()
        self.last_stalled = self.stalled.clone()
        self.last_violated = self.violated.clone()
        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _palms(self):
        pos = self.robot.get_links_pos(self.hands_links_idx)
        quat = self.robot.get_links_quat(self.hands_links_idx)
        return pos + transform_by_quat(self.palm_offset.expand(pos.shape), quat)

    def _update_grip(self):
        """握る・離すの判定。"""
        gc = self.env_cfg["grip"]
        lc = self.env_cfg["ladder"]
        _, _, kmin, dmin, _ = self._ladder_coords(self.palms_pos)
        self.palm_rung_dist, self.palm_rung_idx = dmin, kmin
        command = self.actions[:, self.num_joints :] > 0.0
        within = (dmin < gc["grab_distance"]) & (torch.abs(self.palms_pos[:, :, 1]) < lc["rung_width"] / 2)
        # v11a：手が実際に何かに触れているときだけ握れる（触れていないものは握れない。握った位置は今の手のひらの位置のまま）
        min_contact = gc.get("min_contact_force")
        if min_contact is not None:
            hand_force = torch.norm(self.contact_forces[:, self.hands_links_idx], dim=-1)
            within &= hand_force > min_contact
        # v13：横棒が手のひら側にあるときだけ握れる（手の甲や側面で触れていても握れない）
        #   手のひらの向きは、手の部品の座標の -z 方向（指が曲がる側）。手のひらの中心点から一番近い横棒の中心線への向きが、
        #   手のひらの向きとなす角の cos が palm_facing_cos より大きければ「手のひら側」
        palm_cos = gc.get("palm_facing_cos")
        if palm_cos is not None:
            rung_pt = self.rung_s[kmin][..., None] * self.ladder_u[:, None, :]
            rung_pt[..., 1] = self.palms_pos[..., 1]
            hand_quat = self.robot.get_links_quat(self.hands_links_idx)
            to_rung = transform_by_quat(rung_pt - self.palms_pos, inv_quat(hand_quat))
            facing = -to_rung[..., 2] / torch.norm(to_rung, dim=-1).clamp(min=1e-6)
            within &= facing > palm_cos
        # v13b：横棒を上から握るときだけ握れる（手のひらが下向き寄り：手のひらの向きと真下のなす角の cos が palm_down_cos より大きい）
        #   手のひら側の条件と合わせると、横棒が手のひらの下にある＝上から握っていることになる
        palm_down = gc.get("palm_down_cos")
        if palm_down is not None:
            hand_quat = self.robot.get_links_quat(self.hands_links_idx)
            palm_normal = transform_by_quat(torch.tensor([0.0, 0.0, -1.0], device=gs.device).expand(hand_quat.shape[:-1] + (3,)), hand_quat)
            within &= -palm_normal[..., 2] > palm_down
        # v15b：強制的に離した手は、しばらく握れない（すぐ同じ段を握り直して、片手で支える練習を避けないように）
        within &= self.release_block == 0
        self.release_block = (self.release_block - 1).clamp(min=0)
        # v15d：横棒を下からつかめない：握る瞬間の手のひらの中心が、横棒の中心より min_palm_height_above_rung [m] 以上
        #   下にあったら握れない。v15c では、手を上の段へ移すときに、手のひらを少し上（水平から 25〜28°）に向けて横棒の
        #   下に当てて（中心から約 6 cm 下）握っていた。手の向きの条件（palm_down_cos・palm_facing_cos）だけでは防げなかった
        min_dz = gc.get("min_palm_height_above_rung")
        if min_dz is not None:
            near_z = self.rung_s[kmin] * torch.sin(self.ladder_angle)[:, None]  # 一番近い横棒の中心の高さ
            within &= self.palms_pos[:, :, 2] - near_z > min_dz
        new_grip = command & within & ~self.grip
        # 握った位置：その瞬間の手のひらの位置（横棒にめり込まないよう、今の位置のまま保つ）
        self.grip_anchor = torch.where(new_grip[..., None], self.palms_pos, self.grip_anchor)
        self.grip_rung = torch.where(new_grip, kmin, self.grip_rung)
        self.grip = self.grip | new_grip
        # 離す：指令をやめた、または握力の限界を超えて手が大きくずれた（滑って外れた）
        stretch = torch.norm(self.grip_anchor - self.palms_pos, dim=-1)
        self.grip = self.grip & command & (stretch < gc["slip_distance"])
        # v13b：握っている間に手のひらが握った横棒から大きくそれたら外れる（手のひらの向きと横棒への向きのなす角の cos が
        #   palm_release_cos 未満。握った後に手首を回して、手の甲や側面を横棒に当てる握り方を防ぐ）
        release_cos = gc.get("palm_release_cos")
        if release_cos is not None:
            held_pt = self.rung_s[self.grip_rung][..., None] * self.ladder_u[:, None, :]
            held_pt[..., 1] = self.palms_pos[..., 1]
            hand_quat = self.robot.get_links_quat(self.hands_links_idx)
            to_held = transform_by_quat(held_pt - self.palms_pos, inv_quat(hand_quat))
            held_facing = -to_held[..., 2] / torch.norm(to_held, dim=-1).clamp(min=1e-6)
            self.grip = self.grip & (held_facing > release_cos)
        # v15b：ときどき片手を強制的に離す（学習中だけ）。手の番の間、両手で握り足が横棒に乗っているときに、平均
        #   1 / rate_per_s 秒に 1 回、低い方の手（同じ段ならランダム）を離させ、block_s 秒は握れなくする。
        #   片手でも落ちない姿勢と、離れた手で次の段へ手を伸ばす経験を増やす（v15 では両手の力の合計が体重を超えるほど
        #   腕でぶら下がっていて、片手を離すと残りの手が滑って約 0.5 秒で全員落ちた）
        self.forced_releases = torch.zeros_like(self.forced_releases)
        fr = gc.get("forced_release")
        if fr is not None and self.fixed_angle is None:
            ready = self.hand_turn & self.grip.all(dim=1) & self.feet_on_rung.any(dim=1)
            fire = ready & (torch.rand((self.num_envs,), device=gs.device) < fr["rate_per_s"] * self.dt)
            lower_left = (self.grip_rung[:, 0] < self.grip_rung[:, 1]) | (
                (self.grip_rung[:, 0] == self.grip_rung[:, 1]) & (torch.rand((self.num_envs,), device=gs.device) < 0.5)
            )
            which = torch.stack((lower_left, ~lower_left), dim=1) & fire[:, None]
            self.grip = self.grip & ~which
            self.release_block = torch.where(which, int(round(fr["block_s"] / self.dt)), self.release_block)
            self.forced_releases = which.to(gs.tc_float).sum(dim=1)
        # 高い段を初めて握ったか（腕を使うことを促す一度きりの報酬に使う）
        high = self.grip_anchor[:, :, 2] > self.env_cfg["task"]["first_grip_min_height"]
        grip_high = self.grip & high
        self.first_grip = grip_high & ~self.gripped_once
        self.gripped_once |= grip_high

    def _update_task_state(self):
        tc = self.env_cfg["task"]
        h = self.target_height
        # 手と足の順番（v15）：このステップの更新前の、手が握った最高の段 H と足が乗った最高の段 F（床は -1）の差で決める。
        # 差が hand_turn_max_gap 以下なら「手の番」（手を 1 段上げる）、それより大きければ「足の番」（足を 1 段上げる）。
        # 例：手 4 段目・足 床（差 4）→ 足の番 → 足 1 段目（差 3）→ 手の番 → 手 5 段目（差 4）→ 足の番 → ……
        turn_cfg = tc.get("turn_taking")
        H_before = self.hands_best_rung.max(dim=1).values
        F_before = self.feet_best_rung.max(dim=1).values
        if turn_cfg is not None and turn_cfg.get("ignore_base_height", False):
            F_before = self.feet_seen_best.max(dim=1).values  # v15d：腰との高さを問わずに数えた足の段
        if turn_cfg is not None:
            self.hand_turn = (H_before - F_before) <= turn_cfg["hand_turn_max_gap"]
        else:
            self.hand_turn = torch.zeros_like(self.hand_turn)

        # 足が横棒に乗っているか（v6 で厳しくした）：足の裏が横棒の上面の高さにあり、横棒の上に足が掛かっていて、
        # 力がかかっている。v5 までは「横棒の近くで何かに触れている」だけで数えていたため、足を高く蹴り上げて
        # 横棒に下や横から当てるだけで「乗った」ことになっていた
        feet_force = torch.norm(self.contact_forces[:, self.feet_links_idx], dim=-1)
        rung_pos = self.rung_s[None, :, None] * self.ladder_u[:, None, :]  # [n, 段, 3]
        dz = self.feet_pos[:, :, None, 2] - rung_pos[:, None, :, 2]  # [n, 足, 段]：足のリンクの原点と横棒の中心の高さの差
        dx = self.feet_pos[:, :, None, 0] - rung_pos[:, None, :, 0]  # 足のリンクの原点（足首）と横棒の前後の差
        fw = tc.get("foot_on_rung")
        if fw is not None:
            on = (
                (dz > fw["dz_min"]) & (dz < fw["dz_max"]) & (dx > fw["dx_min"]) & (dx < fw["dx_max"])
                & (torch.abs(self.feet_pos[:, :, None, 1]) < self.env_cfg["ladder"]["rung_width"] / 2)
                & (feet_force[:, :, None] > 5.0)
            )
        else:
            # v5 までの設定で再生するとき：その版の（緩い）判定を使う
            _, _, fk_old, fd_old, _ = self._ladder_coords(self.feet_pos)
            on_old = (self.feet_pos[:, :, 2] > tc["foot_off_floor_height"]) & (fd_old < tc["foot_rung_distance"]) & (feet_force > 5.0)
            on = on_old[:, :, None] & (torch.arange(self.num_rungs, device=gs.device) == fk_old[:, :, None])
        self.feet_on_rung = on.any(dim=-1)
        fk = on.to(torch.int64).argmax(dim=-1)
        self.feet_rung_idx = torch.where(self.feet_on_rung, fk, torch.full_like(fk, -1))

        # 到達度（0〜1）：重心の高さ・横棒に乗った足の高さ・握った手の高さ・横棒への手の近さ
        supported = self.grip.any(dim=1) | self.feet_on_rung.any(dim=1)
        self.support_steps = torch.where(supported, self.support_steps + 1, torch.zeros_like(self.support_steps))
        slow = torch.abs(self.base_vel_world[:, 2]) < tc["max_rise_speed"]
        steady = ((self.support_steps >= tc["support_min_steps"]) & slow).to(gs.tc_float)
        p_com = ((self.com_pos[:, 2] - tc["stand_com_height"]) / h.clamp(min=0.05)).clamp(0.0, 1.0) * steady
        foot_h = torch.where(
            self.feet_on_rung, self.rung_s[self.feet_rung_idx.clamp(min=0)] * torch.sin(self.ladder_angle)[:, None], 0.0
        )
        p_feet = (foot_h / h[:, None].clamp(min=0.05)).clamp(0.0, 1.0).mean(dim=1)
        grip_h = torch.where(self.grip, self.grip_anchor[:, :, 2], 0.0).max(dim=1).values
        p_grip = (grip_h / (h + tc["grip_height_above_target"])).clamp(0.0, 1.0)
        p_reach = (1.0 - self.palm_rung_dist.min(dim=1).values / tc["reach_distance"]).clamp(0.0, 1.0)
        w = tc["progress_weights"]
        self.progress = w[0] * p_reach + w[1] * p_grip + w[2] * p_com + w[3] * p_feet
        if len(w) >= 5:
            # 足を持ち上げた高さ（v8）：手で握っている間に、足（足首）を床から持ち上げた高さ。目標の段の高さで 1（頭打ち）。
            # 足が次の段に実際に乗る前の「足を上へ運ぶ途中」にも手がかりを与える
            foot_up = ((self.feet_pos[:, :, 2] - tc["foot_rest_height"]) / h[:, None].clamp(min=0.05)).clamp(0.0, 1.0)
            p_lift = foot_up.max(dim=1).values * self.grip.any(dim=1).to(gs.tc_float)
            self.progress = self.progress + w[4] * p_lift
        # ぶら下がった姿勢から始めたエピソードは、始めた時点の到達度を基準にする（何もせずに報酬が出ないように）
        fresh_hang = self.is_hang & (self.episode_length_buf <= 1)
        if turn_cfg is not None:
            # v15：到達度のうち重心・足の分（重心の高さ・足の段の高さ・足を持ち上げた高さ）は「足の番」の間だけ増やせる。
            # 手の番の間は、それまでの最高記録で頭打ちにする（手を握り直さずに足だけで登っても到達度が上がらない）
            body = w[2] * p_com + w[3] * p_feet + (w[4] * p_lift if len(w) >= 5 else 0.0)
            self.body_best = torch.where(fresh_hang, body, self.body_best)
            self.body_best = torch.where(self.hand_turn, self.body_best, torch.maximum(self.body_best, body))
            self.progress = self.progress - body + torch.minimum(body, self.body_best)
        self.best_progress = torch.where(fresh_hang, self.progress, self.best_progress)
        self.progress_gain = (self.progress - self.best_progress).clamp(min=0.0)
        self.best_progress = torch.maximum(self.best_progress, self.progress)

        # 段ごとの報酬（v5）：足がこのエピソードで初めて 1 つ上の段に乗るたびに数える（最高記録の更新だけ）
        # 途中の姿勢から始めた直後は、そのときの段を基準にする
        self.feet_best_rung = torch.where(fresh_hang[:, None], self.feet_rung_idx, self.feet_best_rung)
        # v6：同じ段に foot_hold_time_s 秒以上乗り続けてから数える（足を蹴り上げて一瞬だけ高い段に掛ける動きを数えない）。
        # 1 回の更新で数えるのは 1 段まで（床から一気に高い段に足を掛けても 1 段分）
        same = (self.feet_rung_idx == self.feet_prev_rung) & (self.feet_rung_idx >= 0)
        self.feet_hold_steps = torch.where(same, self.feet_hold_steps + 1, torch.zeros_like(self.feet_hold_steps))
        self.feet_prev_rung = self.feet_rung_idx.clone()
        held = self.feet_hold_steps * self.dt >= tc.get("foot_hold_time_s", 0.0)
        held_any_height = torch.where(held, self.feet_rung_idx, torch.full_like(self.feet_rung_idx, -1))  # v15d で使う
        # v7：腰より foot_below_base 以上低い段だけ数える（手と同じ高さの段に足を蹴り上げて掛ける動きを数えない）
        below = tc.get("foot_below_base")
        if below is not None:
            rung_z = self.rung_s[self.feet_rung_idx.clamp(min=0)] * torch.sin(self.ladder_angle)[:, None]
            held &= rung_z < self.base_pos[:, 2:3] - below
        held_rung = torch.where(held, self.feet_rung_idx, torch.full_like(self.feet_rung_idx, -1))
        higher = held_rung > self.feet_best_rung
        if turn_cfg is not None and turn_cfg.get("ignore_base_height", False):
            # v15d：手と足の番・順番違反の判定では、足の段を「同じ段に 0.2 秒以上乗った」だけで数える（腰より 0.35 m 以上低い、
            #   という v7 の条件を使わない）。v15c では、しゃがんで（腰との高さの条件を満たさない形で）3 段目に立つと、加点も
            #   減点もされなかったので、手の番のまま足を上げていた。足の段の加点（+3）は、これまでどおり v7 の条件付き
            self.feet_seen_best = torch.where(fresh_hang[:, None], self.feet_rung_idx, self.feet_seen_best)
            self.feet_turn_best = torch.where(fresh_hang, self.feet_rung_idx.max(dim=1).values, self.feet_turn_best)
            new_seen = held_any_height > self.feet_seen_best  # この足が初めて乗った段（腰との高さは問わない）
            beyond = held_any_height > self.feet_turn_best[:, None]  # 足の番に上げた最高の段より上
            # 足の番に上げてよいのは 1 段だけ。ぶら下がった姿勢から始めて、足がまだどの段にも乗っていないときは、どの段でもよい
            first_from_hang = (self.is_hang_only & (self.feet_turn_best < 0))[:, None]
            allowed = ~self.hand_turn[:, None] & ((held_any_height <= self.feet_turn_best[:, None] + 1) | first_from_hang)
            self.out_of_turn = (new_seen & beyond & ~allowed).to(gs.tc_float).sum(dim=1)
            # v15e：順番違反をしたら、そのエピソードを打ち切る（減点だけの v15〜v15d では、成功の報酬の方が大きく、
            #   減点を払って足だけで目標の段まで登っていた）。転倒の減点・到達度の取り消しはしない
            if turn_cfg.get("terminate_on_violation", False):
                self.violated = self.out_of_turn > 0
            legit = new_seen & beyond & allowed
            self.feet_turn_best = torch.maximum(
                self.feet_turn_best, torch.where(legit, held_any_height, torch.full_like(held_any_height, -1)).max(dim=1).values
            )
            self.feet_seen_best = torch.maximum(self.feet_seen_best, held_any_height)
            # 足の段の加点は、足の番に正しく上げた段（とそこまで追いつかせた段）だけ
            higher = higher & (held_rung <= self.feet_turn_best[:, None])
        elif turn_cfg is not None:
            # v15：足が、足の番に上げた最高の段（feet_turn_best）より上の段に乗ったら、足の番なら加点（記録を更新）、
            # 手の番なら減点。もう一方の足をその段まで追いつかせるのは、いつでも加点。
            # （手の番に片足で上げた段は記録に入れないので、もう一方の足を追いつかせても加点されない）
            self.feet_turn_best = torch.where(fresh_hang, self.feet_best_rung.max(dim=1).values, self.feet_turn_best)
            beyond = held_rung > self.feet_turn_best[:, None]
            foot_turn = ~self.hand_turn[:, None]
            # v15c：足の番に上げてよいのは 1 段だけ（それまでの最高の段の 1 つ上。床からなら 1 段目）。それより上は順番違反。
            #   v15b では、足の番のうちに床から 1・2 段目を飛ばして一気に 3 段目に乗り、手を握り直さずに 3 段目まで登っていた。
            #   ぶら下がった姿勢から始めて、足がまだどの段にも乗っていないときは例外（足を下の段へ下ろすだけなので、どの段でもよい）
            if turn_cfg.get("one_rung_per_turn", False):
                first_from_hang = (self.is_hang_only & (self.feet_turn_best < 0))[:, None]
                foot_turn = foot_turn & ((held_rung <= self.feet_turn_best[:, None] + 1) | first_from_hang)
            self.out_of_turn = (higher & beyond & ~foot_turn).to(gs.tc_float).sum(dim=1)
            higher = higher & (~beyond | foot_turn)
            self.feet_turn_best = torch.maximum(
                self.feet_turn_best, torch.where(higher & beyond, held_rung, torch.full_like(held_rung, -1)).max(dim=1).values
            )
        self.new_rungs = higher.to(gs.tc_float).sum(dim=1)
        self.feet_best_rung = torch.maximum(self.feet_best_rung, held_rung)

        # 手の段ごとの報酬（v7）：片手がこのエピソードで初めて 1 つ上の段を握ったとき。もう一方の手が握っているか、
        # 両足が段に乗っているときだけ数える（体を支えたまま、手を離して上の段を握り直す動きを促す）
        hand_rung = torch.where(self.grip, self.grip_rung, torch.full_like(self.grip_rung, -1))
        # 途中の姿勢から始めた直後は、最初に握らせた段を基準にする（握る指令を一瞬やめて握り直しても数えない）
        self.hands_best_rung = torch.where(fresh_hang[:, None], self.grip_rung, self.hands_best_rung)
        other_grip = self.grip.flip(dims=[1])
        supported_hand = other_grip | self.feet_on_rung.all(dim=1, keepdim=True)
        hand_higher = (hand_rung > self.hands_best_rung) & supported_hand
        rewarded_hand = hand_higher
        if turn_cfg is not None:
            # v15：手の最高の段 H を上げるのは「手の番」のときだけ加点（足の番に上げても加点しない。減点もしない）。
            # もう一方の手を H まで追いつかせるのは、いつでも加点
            H_now = torch.where(fresh_hang, self.grip_rung.max(dim=1).values, H_before)
            raise_H = hand_rung > H_now[:, None]
            rewarded_hand = hand_higher & ~(raise_H & ~self.hand_turn[:, None])
        self.new_hand_rungs = rewarded_hand.to(gs.tc_float).sum(dim=1)
        self.hands_best_rung = torch.where(hand_higher, hand_rung, self.hands_best_rung)

        # 手を次の段へ移す途中の報酬（v15）：「手の番」の間、どちらかの手のひらが「今の H の 1 つ上の段」に近づいた度合い
        # （1 − 距離 / reach_distance）の最高記録を更新した分。H が変わったら、そのときの度合いを基準に測り直す
        self.reach_next_gain = torch.zeros_like(self.reach_next_gain)
        if turn_cfg is not None:
            H = self.hands_best_rung.max(dim=1).values
            nxt = (H + 1).clamp(max=self.num_rungs - 1)
            rung_pt = self.rung_s[nxt][:, None, None] * self.ladder_u[:, None, :]  # [n, 1, 3]
            dv = self.palms_pos - rung_pt
            d = torch.sqrt(dv[..., 0] ** 2 + dv[..., 2] ** 2).min(dim=1).values  # 横棒の中心線までの距離（左右は無視）
            p_next = (1.0 - d / tc["reach_distance"]).clamp(0.0, 1.0)
            new_target = (nxt != self.reach_next_rung) | fresh_hang
            self.reach_next_rung = nxt
            self.reach_next_best = torch.where(new_target, p_next, self.reach_next_best)
            active = self.hand_turn & (H >= 0) & (H + 1 < self.num_rungs) & supported
            self.reach_next_gain = torch.where(active, (p_next - self.reach_next_best).clamp(min=0.0), 0.0)
            self.reach_next_best = torch.where(active, torch.maximum(self.reach_next_best, p_next), self.reach_next_best)

        # じっとしたら打ち切り（v5）：到達度の最高記録も足の段も stall_time_s 秒更新されず、目標の段にもいない
        improved = (self.progress_gain > 1e-4) | (self.new_rungs > 0) | (self.reach_next_gain > 1e-4) | (self.new_hand_rungs > 0)
        self.stall_steps = torch.where(improved, torch.zeros_like(self.stall_steps), self.stall_steps + 1)
        stall_limit = tc.get("stall_time_s")
        self.stalled = torch.zeros_like(self.stalled)
        if stall_limit is not None:
            self.stalled = (self.stall_steps * self.dt > stall_limit) & ~self.on_target

        # 目標の段に立っているか：両足が目標の段以上の横棒に乗り、少なくとも片手で握っている
        self.on_target = (
            self.feet_on_rung.all(dim=1) & (self.feet_rung_idx >= self.episode_target[:, None]).all(dim=1) & self.grip.any(dim=1)
        )
        if turn_cfg is not None and turn_cfg.get("success_requires_order", False):
            # v15f：両足が「足の番に正しく上げた段」以下にあるときだけ、目標の段に立ったとみなす。成功の判定は足が触れた瞬間に
            #   成り立つが、順番違反は同じ段に 0.2 秒乗ってから数えるので、v15e では順番を守らずに両足を目標の段に乗せて
            #   成功の報酬を受け取り、その 0.2 秒後に順番違反で打ち切られていた
            self.on_target &= (self.feet_rung_idx <= self.feet_turn_best[:, None]).all(dim=1)
        self.just_succeeded = self.on_target & ~self.succeeded
        self.succeeded |= self.on_target

        # 体のどこかが床かはしごに触れているか、手で握っているか（どれでもない＝跳んでいる）
        self.in_contact = (torch.norm(self.contact_forces, dim=-1) > 1.0).any(dim=1) | self.grip.any(dim=1)
        self.flight_steps = torch.where(self.in_contact, torch.zeros_like(self.flight_steps), self.flight_steps + 1)
        self.has_landed |= self.in_contact
        jumped = (self.flight_steps >= tc["max_flight_steps"]) & self.has_landed

        # はしごの下にもぐり込んだ：足か腰が、はしごの面より向こう側に入った（v2）
        # （v1 では、緩やかなはしごの下に足を入れてしゃがみ、低い段を握ったまま動かない動きを覚えた）
        _, foot_n, _, _, _ = self._ladder_coords(self.feet_pos)
        _, base_n, _, _, _ = self._ladder_coords(self.base_pos[:, None, :])
        self.under_ladder = (foot_n < -tc.get("max_foot_behind", 0.05)).any(dim=1) | (base_n[:, 0] < 0.0)

        # 転倒：腰が低すぎる／大きく傾いた／はしごの下にもぐり込んだ
        self.fell = (
            (self.base_pos[:, 2] < tc["min_base_height"])
            | self.under_ladder
            | jumped
            | (torch.abs(self.base_euler[:, 0]) > tc["max_roll"])
            | (self.base_euler[:, 1] > tc["max_pitch_forward"])
            | (self.base_euler[:, 1] < -tc["max_pitch_back"])
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf, "critic": self.privileged_obs_buf}, batch_size=[self.num_envs])

    def _update_observation(self):
        yaw = self.base_euler[:, 2]
        th = self.ladder_angle
        base_s, base_n, _, _, _ = self._ladder_coords(self.base_pos[:, None, :])
        _, palm_n, _, _, palm_ds = self._ladder_coords(self.palms_pos)
        _, foot_n, _, _, foot_ds = self._ladder_coords(self.feet_pos)
        ladder_obs = torch.cat(
            (
                torch.stack((torch.sin(th), torch.cos(th), torch.sin(yaw), torch.cos(yaw)), dim=1),  # 4
                base_s, base_n,  # 2：腰の位置（はしごに沿った距離・はしごの面からの距離）
                palm_ds, palm_n,  # 4：手のひらから一番近い横棒まで（はしごに沿った方向・面に垂直な方向）
                foot_ds, foot_n,  # 4：足から一番近い横棒まで
                self.grip.to(gs.tc_float),  # 2：握っているか
                self.palms_pos[:, :, 1],  # 2：手のひらの横の位置（横棒の端から外れないように）
            ),
            dim=1,
        )
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 23
                self.dof_vel * self.obs_scales["dof_vel"],  # 23
                self.actions,  # 25
                ladder_obs,  # 18
            ),
            dim=-1,
        )
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_pos[:, 2:3],  # 1
                self.feet_on_rung.to(gs.tc_float),  # 2
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

        n_done = mask.sum()
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            mean = torch.where(n_done > 0, value[mask].sum() / n_done.clamp(min=1), 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            value.masked_fill_(mask, 0.0)

        # カリキュラムの更新（評価時は傾きを固定する。復習のエピソードは段階の更新に使わない）
        if not all_envs and self.fixed_angle is None:
            self._update_curriculum(mask & ~self.is_review & ~self.is_hang)
        cc = self.curriculum_cfg
        review = torch.zeros_like(mask)
        override = torch.full_like(self.ladder_angle, float("nan"))
        if self.fixed_angle is None and cc.get("review_prob", 0.0) > 0.0:
            review = mask & (torch.rand((self.num_envs,), device=gs.device) < cc["review_prob"])
            review_levels = torch.randint(0, self.num_levels, (self.num_envs,), device=gs.device)
            override = torch.where(review, cc["angle_min"] + review_levels.to(gs.tc_float) * cc["angle_step"], override)
        self.angle_override = torch.where(mask, override, self.angle_override)
        self.is_review = torch.where(mask, review, self.is_review)
        self._update_ladder(None if all_envs else mask)
        angles = cc["angle_min"] + self.levels.to(gs.tc_float) * cc["angle_step"]
        self.extras["episode"]["Curriculum/mean_angle"] = angles.mean()
        self.extras["episode"]["Curriculum/max_angle"] = angles.max()
        self.extras["episode"]["Curriculum/assist"] = torch.tensor(self.assist_scale)
        self.extras["episode"]["Curriculum/success_rate"] = torch.tensor(self.success_rate_ema)
        self.extras["episode"]["Curriculum/mean_target_rung"] = (self.target_rung_env + 1).to(gs.tc_float).mean()  # 下から数えた段

        # ロボットをはしごの手前に置き直す
        rc = self.env_cfg["reset"]
        spawn_x = -gs_rand(*rc["distance_range"], (self.num_envs,))
        spawn_y = gs_rand(-rc["lateral_range"], rc["lateral_range"], (self.num_envs,))
        spawn_yaw = gs_rand(-rc["yaw_range"], rc["yaw_range"], (self.num_envs,))
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
        # 一部の環境は、はしごの高い段を両手で握ってぶら下がった姿勢から始める（v3。逆向きのカリキュラム）
        #   体をはしごと平行に傾け、手のひらが横棒の少し手前に来るように腰の位置を決める。始めた時点で両手は握っている
        hang = torch.zeros_like(mask)
        # 学習中はランダムに混ぜる。評価中は set_eval_start で指定したときだけ使う
        use_hang = bool(self.hang_cfg) and (self.fixed_angle is None or self.eval_start == "hang")
        use_stand = bool(self.stand_cfg) and (self.fixed_angle is None or self.eval_start == "stand")
        if use_hang:
            hc = self.hang_cfg
            if self.eval_start == "hang":
                hang = mask.clone()
            else:
                hang = mask & ~review & (torch.rand((self.num_envs,), device=gs.device) < hc["prob"])
            m = torch.randint(hc["rungs"][0], hc["rungs"][1] + 1, (self.num_envs,), device=gs.device)
            if self.eval_hang_rung is not None:
                m = torch.full_like(m, int(self.eval_hang_rung))
            lean = math.pi / 2 - self.ladder_angle  # はしごと平行になるよう前に傾ける角度
            hang_quat = torch.stack(
                (torch.cos(lean / 2), torch.zeros_like(lean), torch.sin(lean / 2), torch.zeros_like(lean)), dim=1
            )
            offset = transform_by_quat(self.hang_palm_offset.mean(dim=0).expand(self.num_envs, -1), hang_quat)
            target = self.rung_s[m][:, None] * self.ladder_u + hc["standoff"] * self.ladder_nrm
            target[:, 1] = 0.0
            hang_pelvis = target - offset
            hang_qpos = torch.cat((hang_pelvis, hang_quat, self.hang_dof_pos_all.expand(self.num_envs, -1)), dim=1)
            qpos = torch.where(hang[:, None], hang_qpos, qpos)
            hang_anchor = hang_pelvis[:, None, :] + transform_by_quat(
                self.hang_palm_offset.expand(self.num_envs, -1, -1), hang_quat[:, None, :].expand(-1, 2, -1)
            )
        # 一部の環境は、床に立ったまま、肩より少し上の段を両手で握った姿勢から始める（v4。床からはしごへ移る練習）
        #   体はまっすぐ立てたまま、手のひらが横棒の少し手前に来る位置に立たせる。足がはしごの下に入ってしまう
        #   緩やかなはしごや、腰が高く浮きすぎる場合は、ふつうの姿勢から始める
        stand = torch.zeros_like(mask)
        if use_stand:
            sc = self.stand_cfg
            sin_t, cos_t = torch.sin(self.ladder_angle), torch.cos(self.ladder_angle)
            off = self.stand_palm_offset.mean(dim=0)
            palm_z = sc["height"] + off[2]
            lc = self.env_cfg["ladder"]
            m_s = torch.round((palm_z / sin_t - lc["rung_first"]) / lc["rung_spacing"]).long().clamp(1, self.num_rungs - 1)
            target = self.rung_s[m_s][:, None] * self.ladder_u + sc["standoff"] * self.ladder_nrm
            stand_pelvis = target - off
            stand_pelvis[:, 1] = 0.0
            stand_pelvis[:, 2] = stand_pelvis[:, 2].clamp(min=sc["height"])  # 脚が床にめり込まないように
            foot_n = -sin_t * stand_pelvis[:, 0] + cos_t * 0.05
            valid = (foot_n > -0.03) & (stand_pelvis[:, 2] < sc["height"] + 0.15)
            if self.eval_start == "stand":
                stand = mask & valid
            else:
                stand = mask & ~review & ~hang & valid & (torch.rand((self.num_envs,), device=gs.device) < sc["prob"])
            stand_qpos = torch.cat(
                (stand_pelvis, torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device).expand(self.num_envs, -1),
                 self.stand_dof_pos_all.expand(self.num_envs, -1)),
                dim=1,
            )
            qpos = torch.where(stand[:, None], stand_qpos, qpos)
            stand_anchor = stand_pelvis[:, None, :] + self.stand_palm_offset[None]
            _, _, _, d_anchor, _ = self._ladder_coords(stand_anchor)
            stand_grip = stand[:, None] & (d_anchor < self.env_cfg["grip"]["grab_distance"])
        self.is_hang = torch.where(mask, hang | stand, self.is_hang)
        self.is_hang_only = torch.where(mask, hang, self.is_hang_only)
        # 今のエピソードの目標の段：床からの開始は環境ごとの目標の段、ぶら下がり・床に立って握った姿勢からの開始と
        # 評価のときは一番上の目標（task.target_rung）
        k_max = self.env_cfg["task"]["target_rung"]
        ep_target = torch.where(self.is_hang, torch.full_like(self.target_rung_env, k_max), self.target_rung_env)
        if self.fixed_angle is not None:
            ep_target = torch.full_like(self.target_rung_env, k_max)
        if use_hang and self.hang_cfg.get("target_offset") is not None:
            # v15：ぶら下がり開始の目標は「握った段 + target_offset」（-1 ならすぐ下の段）。手を動かさずに足を下ろすだけでは
            # 届かない段にして、手を上の段へ握り直す練習をさせる（v14 までは上限の段が目標で、足を下ろすだけで成功できた）
            hang_target = (m + self.hang_cfg["target_offset"]).clamp(0, self.num_rungs - 1)
            ep_target = torch.where(hang, hang_target, ep_target)
        self.episode_target = torch.where(mask, ep_target, self.episode_target)
        self.target_height = self.rung_s[self.episode_target] * torch.sin(self.ladder_angle)
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
        for buf in (self.succeeded, self.on_target, self.just_succeeded, self.fell, self.has_landed):
            buf.masked_fill_(mask, False)
        for buf in (self.grip, self.first_grip, self.gripped_once, self.feet_on_rung):
            buf.masked_fill_(m1, False)
        self.grip_force.masked_fill_(mask[:, None, None], 0.0)
        if use_hang:
            self.grip |= hang[:, None]
            self.gripped_once |= hang[:, None]  # 最初から握っている分には、初めて握った報酬を出さない
            self.grip_anchor = torch.where(hang[:, None, None], hang_anchor, self.grip_anchor)
            self.grip_rung = torch.where(hang[:, None], m[:, None].expand(-1, 2), self.grip_rung)
        if use_stand:
            self.grip |= stand_grip
            self.gripped_once |= stand[:, None]
            self.grip_anchor = torch.where(stand_grip[..., None], stand_anchor, self.grip_anchor)
            self.grip_rung = torch.where(stand[:, None], m_s[:, None].expand(-1, 2), self.grip_rung)
        self.feet_rung_idx.masked_fill_(m1, -1)
        self.in_contact.masked_fill_(mask, True)
        self.support_steps.masked_fill_(mask, 0)
        self.stall_steps.masked_fill_(mask, 0)
        self.stalled.masked_fill_(mask, False)
        self.violated.masked_fill_(mask, False)
        self.new_rungs.masked_fill_(mask, 0.0)
        self.feet_best_rung.masked_fill_(m1, -1)
        self.feet_hold_steps.masked_fill_(m1, 0)
        self.feet_prev_rung.masked_fill_(m1, -1)
        self.hands_best_rung.masked_fill_(m1, -1)
        self.new_hand_rungs.masked_fill_(mask, 0.0)
        for buf in (self.out_of_turn, self.body_best, self.reach_next_best, self.reach_next_gain):
            buf.masked_fill_(mask, 0.0)
        self.reach_next_rung.masked_fill_(mask, -1)
        self.feet_turn_best.masked_fill_(mask, -1)
        self.feet_seen_best.masked_fill_(m1, -1)
        self.release_block.masked_fill_(m1, 0)
        self.forced_releases.masked_fill_(mask, 0.0)
        self.hand_turn.masked_fill_(mask, False)
        self.flight_steps.masked_fill_(mask, 0)
        self.reset_buf.masked_fill_(mask, True)

    def reset(self):
        self._reset_idx()
        self.palms_pos = self._palms()
        _, _, kmin, dmin, _ = self._ladder_coords(self.palms_pos)
        self.palm_rung_dist, self.palm_rung_idx = dmin, kmin
        self._update_observation()
        return self.get_observations()

    # ----------------------------------------------------------------------
    # 報酬関数
    # ----------------------------------------------------------------------
    # --- タスク（sparse：dt を掛けない） ---
    def _reward_progress(self):
        return self.progress_gain

    def _reward_success(self):
        return self.just_succeeded.to(gs.tc_float)

    def _reward_new_hand_rung(self):
        # 片手がこのエピソードで初めて 1 つ上の段を握ったとき（v7）
        return self.new_hand_rungs

    def _reward_new_rung(self):
        # 足がこのエピソードで初めて 1 つ上の段に乗ったとき（段ごと・足ごと。v5）
        return self.new_rungs

    def _reward_out_of_turn(self):
        # 手の番なのに、足の最高の段を上げた（v15）
        return self.out_of_turn

    def _reward_hand_reach(self):
        # 手の番の間、手のひらが次の段に近づいた度合いの最高記録の更新量（v15）
        return self.reach_next_gain

    def _reward_first_grip(self):
        # 高い段の横棒を初めて握ったとき（片手ごとに一度だけ）
        return self.first_grip.to(gs.tc_float).sum(dim=1)

    def _reward_termination(self):
        # 転倒して終わったとき。このエピソードで得た到達度の報酬も取り消す
        fell = (self.fell & ~self.time_out_buf).to(gs.tc_float)
        refund = self.best_progress * self.reward_cfg["sparse_scales"]["progress"] / abs(self.reward_scales["termination"])
        return fell * (1.0 + refund)

    # --- タスク（dense：毎ステップ） ---
    def _reward_on_target(self):
        # 目標の段に両足で立ち、手で握っている間
        return self.on_target.to(gs.tc_float)

    def _reward_alive(self):
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _reward_body_contact(self):
        # 手足以外（胴体・腰・脚・腕）が、はしごや床を押している力（体重比）（v2。胸をはしごにもたれかけて休む動きを抑える）
        force = torch.norm(self.contact_forces[:, self.body_links_idx], dim=-1).sum(dim=1)
        return force / (self.robot_mass * 9.81)

    def _reward_grip_strain(self):
        # 握力の限界に近い力で手を引きつけている（腕でぶら下がるより、足で体重を支えることを促す）
        return (torch.norm(self.grip_force, dim=-1) / self.env_cfg["grip"]["max_force"]).square().sum(dim=1)

    def _reward_grip_load(self):
        # 足が横棒に乗っている間、両手で引きつけている力の合計（体重比）（v15b。足で体重を支えることを促す）
        # 片足でも乗っていれば数える（両足のときだけにすると、片足を浮かせてこの減点と強制的に手を離す練習を避けられる）
        weight = float(self.link_masses.sum()) * 9.81
        load = torch.norm(self.grip_force, dim=-1).sum(dim=1) / weight
        return load * self.feet_on_rung.any(dim=1).to(gs.tc_float)

    # --- 動きの質（dense） ---
    def _reward_flight(self):
        return (~self.in_contact).to(gs.tc_float)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_yaw_rate(self):
        return torch.square(self.robot.get_ang()[:, 2])

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
        excess = torch.norm(self.contact_forces, dim=-1) - self.reward_cfg["max_contact_force"]
        return torch.sum(excess.clip(min=0.0), dim=1)
