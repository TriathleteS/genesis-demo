"""Unitree G1（23 関節）が、傾いたはしごを手と足を使って登る方策を強化学習（PPO）で学習するスクリプト。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_ladder_train.py                          # 4096 環境 × 3000 イテレーション
    uv run python locomotion/g1_ladder_train.py --resume 2999 --max-iterations 1000
                                                                         # model_2999.pt から 1000 イテレーション続ける
    uv run python locomotion/g1_ladder_train.py --init-from g1-ladder:2999 -e g1-ladder-v2 --max-iterations 1000
                                                                         # 別の実験の学習済みモデルから、今の設定で学習する

学習結果は logs/<実験名>/ に保存される（既定の実験名は g1-ladder-v2。1 回目は g1-ladder）。
--resume を付けずに同じ実験名で実行すると、前回の logs/<実験名>/ は削除されて上書きされる点に注意。

学習中に TensorBoard で見るとよい項目:
    Curriculum/mean_angle   : 全環境のはしごの傾きの平均 [deg]（上がっていけば、急なはしごを登れるようになっている）
    Curriculum/success_rate : 目標の段まで登れたエピソードの割合（移動平均）
    Episode/rew_first_grip  : 高い段の横棒を握った回数に比例（手を使い始めたか）
    Episode/rew_on_target   : 目標の段に立ち、握り続けている時間に比例
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_ladder_env import G1LadderEnv


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            # よじ登り（v5〜v9）では 0.01 で行動のばらつきが増え続けた。はしご v1 の 0.003 では逆に
            # 800 イテレーションで 0.39 まで下がり、試行錯誤が止まったので、その間の 0.005 にする（v2）
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint",
    ]  # fmt: skip
    env_cfg = {
        "joint_names": joint_names,
        # 行動が 0 のときの目標姿勢 [rad]：膝を軽く曲げて立ち、腕は肘を曲げて体の前に構える
        "default_joint_angles": {
            "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.25, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.9, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.25, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.9, "right_wrist_roll_joint": 0.0,
        },  # fmt: skip
        "base_link_name": "pelvis",
        "feet_link_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "ankle_link_names": ["left_ankle_pitch_link", "right_ankle_pitch_link"],
        "hand_link_names": ["left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"],
        "palm_offset": (0.08, 0.0, 0.0),  # 手のリンクの原点（手首）から手のひらの中心まで [m]（手のリンクの座標系）
        # PD 制御 [N·m/rad]（関節名の部分一致で指定）
        "stiffness": {"hip": 100, "knee": 150, "ankle": 40, "waist": 150, "shoulder": 40, "elbow": 40, "wrist": 20},
        "damping": {"hip": 2, "knee": 4, "ankle": 2, "waist": 4, "shoulder": 1, "elbow": 1, "wrist": 0.5},
        # 行動 1 あたりの目標角度の変化 [rad]（腕は大きく動かす必要があるので大きめ）
        "action_scale": {"hip": 0.25, "knee": 0.25, "ankle": 0.25, "waist": 0.25, "shoulder": 0.5, "elbow": 0.5,
                         "wrist": 0.25},
        "clip_actions": 100.0,
        # シミュレーション
        "dt": 0.02,
        "substeps": 4,
        "self_collision": False,
        "max_collision_pairs": 100,
        "episode_length_s": 15.0,
        # はしご：横棒の間隔 0.25 m（はしごに沿った距離）、一番下の段は根元から 0.25 m
        "ladder": {
            "num_rungs": 8,
            "rung_first": 0.25,
            "rung_spacing": 0.25,
            "rung_width": 0.6,  # 横棒の長さ [m]
            "rung_thickness": 0.04,  # 横棒の太さ（正方形の断面の一辺）[m]
        },
        # 握る動作（簡略化）：手のひらを、握った位置にばねで引きつける
        "grip": {
            "grab_distance": 0.07,  # 手のひらが横棒の中心線からこの距離 [m] 以内なら握れる
            "stiffness": 3000.0,  # [N/m]
            "damping": 60.0,  # [N·s/m]
            "max_force": 250.0,  # 握力の上限 [N]（片手。体重 32 kg の約 8 割）
            "slip_distance": 0.12,  # 手のひらが握った位置からこれ以上ずれたら、滑って外れたとみなす [m]
        },
        # リセット時のロボットの置き方
        "reset": {
            "distance_range": (0.3, 0.45),  # 腰からはしごの根元までの距離 [m]
            "lateral_range": 0.05,  # 左右のずれ [m]
            "yaw_range": 0.1,  # 向きのずれ [rad]
            "height": 0.8,  # 腰の高さ [m]
        },
        # タスクの判定
        "task": {
            "target_rung": 2,  # 両足でこの段（下から 3 段目、はしごに沿って 0.75 m）以上に立てたら成功
            # 到達度の内訳：横棒への手の近さ・握った手の高さ・重心の上がった割合・横棒に乗った足の高さ
            "progress_weights": (0.05, 0.2, 0.5, 0.25),
            "reach_distance": 0.5,  # 手のひらが横棒からこの距離 [m] 以内に近づくと「手の近さ」が増え始める
            "grip_height_above_target": 1.0,  # 握った手の高さが「目標の段の高さ + これ」で 1 [m]
            "first_grip_min_height": 0.7,  # この高さ [m] 以上の横棒を初めて握ったときだけ報酬
            "stand_com_height": 0.68,  # 立っているときの全身の重心の高さ [m]
            "support_min_steps": 5,  # 握るか足を横棒に乗せて 0.1 秒たってから重心の高さを数える
            "max_rise_speed": 0.8,  # 胴体の上下の速さがこれ [m/s] 未満のときだけ重心の高さを数える
            "foot_off_floor_height": 0.08,  # 足のリンクがこれより高ければ床から離れている [m]
            "foot_rung_distance": 0.12,  # 足のリンクが横棒の中心線からこの距離 [m] 以内なら横棒に乗っている
            "min_base_height": 0.4,  # 腰がこれより低くなったら転倒 [m]
            "max_roll": 1.0,
            "max_pitch_forward": 1.0,
            "max_pitch_back": 0.8,  # 後ろに倒れた（はしごから体が離れた）とみなす角度 [rad]
            "max_flight_steps": 5,  # 体のどこも触れず、握ってもいない状態が 0.1 秒続いたら失敗（跳躍の禁止）
            # 足がはしごの面よりこの距離 [m] 以上向こう側に入るか、腰がはしごの面より向こう側に入ったら失敗（v2。はしごの下にもぐり込む動きの禁止）
            "max_foot_behind": 0.05,
        },
    }
    obs_cfg = {
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales": {"dof_pos": 0.01, "dof_vel": 1.5, "ang_vel": 0.2, "gravity": 0.05, "ladder": 0.02},
    }
    reward_cfg = {
        "soft_dof_pos_limit": 0.9,
        "max_contact_force": 400.0,
        # 1 エピソードに一度だけ（または到達度の更新時だけ）与える報酬
        "sparse_scales": {
            "progress": 20.0,  # 最高到達度の更新量 × 20（目標まで登ると合計 20）
            "success": 10.0,  # 初めて目標の段に立てたとき
            "first_grip": 2.0,  # 高い段の横棒を初めて握ったとき（片手ごと）
            "termination": -2.0,  # 転倒したとき（このエピソードで得た到達度の報酬も取り消す）
        },
        # 毎ステップ与える報酬（1 秒あたりの値。内部で dt を掛ける）
        "dense_scales": {
            "on_target": 10.0,  # 目標の段に立ち、握り続けている間
            "alive": 0.2,  # v1 の 0.5 では、転ばずにじっとしているだけで得をしたので下げる（v2）
            "body_contact": -5.0,  # 手足以外がはしごや床を押す力（体重比）。v1 では胸を緩やかなはしごにもたれかけて休んでいた（v2）
            "grip_strain": -0.5,  # 握力の限界に近い力で引きつけている（腕だけでぶら下がる動きを抑える）
            "flight": -5.0,
            "lin_vel_z": -1.0,
            "yaw_rate": -0.3,
            "action_rate": -0.01,
            "dof_acc": -2.5e-7,
            "dof_vel": -1e-3,
            "dof_pos_limits": -5.0,
            "contact_force": -5e-3,
        },
    }
    curriculum_cfg = {
        # はしごの傾き [deg]：60°（急な階段くらい）から 90°（垂直）まで 2.5° 刻み
        "angle_min": 60.0,
        "angle_max": 90.0,
        "angle_step": 2.5,
        "init_max_level": 2,  # 最初は 60〜65° からランダムに始める
        "level_down_progress": 0.3,  # 最高到達度がこれ未満で終わったら段階を下げる
        "top_levels_resample": 4,  # 最上段を超えた環境は、上から 4 段の中で選び直す
        "review_prob": 0.2,  # エピソードの 2 割は、段階と関係なくランダムな傾きで練習する（段階の更新には使わない）
        # 補助の力：はしごに手か足が掛かっている間、体重 × assist_init の上向きの力を腰にかける
        "assist_init": 0.4,
        "assist_anneal_steps": 1500 * 24,  # 1500 イテレーションかけて 0 まで下げる
    }
    return env_cfg, obs_cfg, reward_cfg, curriculum_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-ladder-v2", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="並列に動かす環境の数")
    parser.add_argument("--max-iterations", type=int, default=3000, help="学習のイテレーション数")
    parser.add_argument("--resume", type=int, default=None, help="このイテレーションのチェックポイントから学習を再開する")
    parser.add_argument(
        "--init-from", type=str, default=None, metavar="EXP:ITER",
        help="別の実験の学習済みモデルから学習を始める（例：g1-ladder:2999）。報酬などの設定は今回のものを使う",
    )
    parser.add_argument("--seed", type=int, default=1, help="乱数のシード")
    parser.add_argument("--cpu", action="store_true", help="CPU で実行する（動作確認用。学習には GPU が必要）")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    if args.resume is None:
        env_cfg, obs_cfg, reward_cfg, curriculum_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, curriculum_cfg, train_cfg], f)
    else:
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            env_cfg, obs_cfg, reward_cfg, curriculum_cfg, train_cfg = pickle.load(f)

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env = G1LadderEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        curriculum_cfg=curriculum_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(os.path.join(log_dir, f"model_{args.resume}.pt"))
        env.common_step_counter = args.resume * train_cfg["num_steps_per_env"]
    elif args.init_from:
        src_exp, src_iter = args.init_from.rsplit(":", 1)
        runner.load(os.path.join("logs", src_exp, f"model_{src_iter}.pt"), map_location=gs.device)
        print(f"{src_exp} の model_{src_iter}.pt から学習を始めます。")
        env.common_step_counter = curriculum_cfg["assist_anneal_steps"]

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
