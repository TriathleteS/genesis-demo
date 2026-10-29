"""Unitree G1（23 関節）が台によじ登る方策を強化学習（PPO）で学習するスクリプト。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_climb_train.py                         # 4096 環境 × 3000 イテレーション
    uv run python locomotion/g1_climb_train.py --replay locomotion/climb_start_states.pt
                                                                       # 保存しておいた途中の姿勢から始めるエピソードを 3 割混ぜる
    uv run python locomotion/g1_climb_train.py --resume 2999 --max-iterations 1000
                                                                       # model_2999.pt から 1000 イテレーション続ける

学習結果は logs/<実験名>/ に保存される（既定の実験名は g1-climb-v7。以前の版は g1-climb, g1-climb-v2〜v6）。
--resume を付けずに同じ実験名で実行すると、前回の logs/<実験名>/ は削除されて上書きされる点に注意。

学習中に TensorBoard で見るとよい項目:
    Curriculum/mean_height  : 全環境の台の高さの平均（上がっていけば、高い台に登れるようになっている）
    Curriculum/success_rate : 登り切れたエピソードの割合（移動平均）
    Curriculum/assist       : 補助の力の強さ（体重に対する割合。学習の前半で 0 まで下がる）
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

from g1_climb_env import G1ClimbEnv


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
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
        "num_actions": len(joint_names),
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
        "hand_link_names": ["left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"],
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
        "episode_length_s": 10.0,
        # 台：x 方向の奥行き 1.0 m、幅 2.0 m、高さ 1.5 m の直方体を、上面が目標の高さになるよう床に埋めて置く
        "box_size": (1.0, 2.0, 1.5),
        # リセット時のロボットの置き方
        "reset": {
            "distance_range": (0.25, 0.4),  # 腰から台の正面までの距離 [m]（歩く必要がない距離から始める）
            "lateral_range": 0.05,  # 左右のずれ [m]
            "yaw_range": 0.1,  # 向きのずれ [rad]
            "height": 0.8,  # 腰の高さ [m]
        },
        # タスクの判定
        "task": {
            "stand_height": 0.78,  # 立っているときの腰の高さ [m]
            # 到達度の内訳：足を上げた高さ・全身の重心の上がった割合・台に乗った足の数・重心が台の上へ前進した割合
            # （足を上げた高さは最初の手がかり。これがないと台に近づく理由がなく、何もしないまま学習が止まる）
            # （重心の前進は v7 で追加。片足を台に乗せたまま、台の手前で踏ん張り続けるのを防ぐ）
            "progress_weights": (0.1, 0.5, 0.2, 0.2),
            "com_fwd_start": -0.3,  # 重心の前進を数え始める位置 [m]（台の正面が x = 0。台の手前に立っているときの重心の位置）
            "com_fwd_target": 0.2,  # 重心がここまで台の上に進めば前進の割合が 1 [m]
            "foot_near_box_x": 0.3,  # 台の正面からこの距離 [m] 以内の足だけ「足を上げた高さ」に数える
            "stand_com_height": 0.68,  # 立っているときの全身の重心の高さ [m]
            "support_min_steps": 5,  # 手か足が台に触れ続けてこのステップ数（0.1 秒）たってから重心の高さを数える
            "max_rise_speed": 0.8,  # 胴体の上下の速さがこれ [m/s] 未満のときだけ重心の高さを数える
            "foot_rest_height": 0.035,  # 床に立っているときの足のリンクの高さ [m]
            "on_box_margin_x": 0.03,  # 台の縁からこれ以上内側にある足・手を「台の上」とみなす [m]
            "upright_tilt": 0.35,  # 台の上で「立っている」とみなす傾きの上限（重力方向の横成分）
            "min_height_on_ground": 0.35,  # 台の手前で腰がこれより低くなったら転倒 [m]
            "min_height_over_box": 0.08,  # 台の上で腰が上面からこれより低くなったら転倒 [m]
            "max_roll": 1.2,  # 横に倒れたとみなす角度 [rad]
            "max_pitch_forward": 0.9,  # 前に倒れたとみなす角度 [rad]（約 52 度。台に覆いかぶさる姿勢を防ぐ）
            "max_pitch_back": 1.0,  # 後ろに倒れたとみなす角度 [rad]
            "max_flight_steps": 3,  # 体のどこも床や台に触れていない状態がこのステップ数（0.06 秒）続いたら失敗（跳躍の禁止）
            "left_box_height": 0.4,  # 台に登った後、腰が上面からこれより低くなり両足が台から離れたら失敗 [m]
        },
    }
    obs_cfg = {
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales": {"dof_pos": 0.01, "dof_vel": 1.5, "ang_vel": 0.2, "gravity": 0.05, "box": 0.02},
    }
    reward_cfg = {
        "soft_dof_pos_limit": 0.9,
        "max_contact_force": 400.0,  # [N] これを超える接触力にペナルティ
        "heading_free": 0.5,  # 台の方向からこの角度 [rad]（約 30 度）までの向きの変化はペナルティなし
        "hands_touch_min_height": 0.5,  # 手が台に触れた報酬は、この高さ [m] 以上の台でだけ与える
        # 1 エピソードに一度だけ（または到達度の更新時だけ）与える報酬
        "sparse_scales": {
            "progress": 20.0,  # 最高到達度の更新量 × 20（登り切ると合計 20）
            "success": 10.0,  # 初めて台の上に立てたとき
            "hands_touch": 1.0,  # 手が初めて台の上に触れたとき（片手ごと）
            "termination": -2.0,  # 転倒したとき（このエピソードで得た到達度の報酬も取り消す）
        },
        # 毎ステップ与える報酬（1 秒あたりの値。内部で dt を掛ける）
        # 注意：「手を台についている」ことへの毎ステップの報酬は入れない。入れると、台に手をついて
        # うつ伏せにもたれかかったまま時間切れまで報酬を稼ぐ抜け道を覚えてしまう。
        # 「転ばずにいる」報酬（alive）は、前に 52 度以上傾いたら転倒とみなす判定と組み合わせて使う
        "dense_scales": {
            "on_top": 10.0,  # 台の上に立ち続けている間（登った後に台の上で安定して立つことを重視する）
            "alive": 0.5,
            "trailing_foot": 1.0,  # 片足が台の上にあるとき、もう一方の足を上げた割合
            "lean_back": -2.0,  # 後ろへののけぞり
            "heading": 0.0,  # 無効（v6 では跳躍を失敗にしたので不要と判断）。使うときは -2.0 など：台の方向から heading_free [rad] 以上向きを変えた分の 2 乗
            "upright_on_top": -2.0,
            "action_rate": -0.01,
            "dof_acc": -2.5e-7,
            "dof_vel": -1e-3,
            "dof_pos_limits": -5.0,
            "contact_force": -5e-3,
            "flight": -5.0,  # 体のどこも床や台に触れていない（跳んでいる）間
            "lin_vel_z": -1.0,  # 胴体の上下の速さの 2 乗
        },
    }
    curriculum_cfg = {
        # 台の高さの段階：0.1 m から 1.0 m まで 0.05 m 刻み
        # （v7 で下限を 0.2 m から 0.1 m に下げた。0.1 m なら歩いて乗れるので、両足で台に乗る経験から始められる）
        "height_min": 0.1,
        "height_max": 1.0,
        "height_step": 0.05,
        "init_max_level": 4,  # 最初は 0.1〜0.3 m からランダムに始める
        "level_down_progress": 0.5,  # 最高到達度がこれ未満で終わったら段階を下げる
        "top_levels_resample": 5,  # 最上段を超えた環境は、上から 5 段の中で選び直す
        # 補助の力：手か足が台の上に触れている間、体重 × assist_init の上向きの力を腰にかける
        "assist_init": 0.4,
        "assist_anneal_steps": 1500 * 24,  # 1500 イテレーションかけて 0 まで下げる
    }
    return env_cfg, obs_cfg, reward_cfg, curriculum_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-climb-v7", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="並列に動かす環境の数")
    parser.add_argument("--max-iterations", type=int, default=3000, help="学習のイテレーション数")
    parser.add_argument("--resume", type=int, default=None, help="このイテレーションのチェックポイントから学習を再開する")
    parser.add_argument("--seed", type=int, default=1, help="乱数のシード")
    parser.add_argument("--cpu", action="store_true", help="CPU で実行する（動作確認用。学習には GPU が必要）")
    parser.add_argument(
        "--replay", type=str, default=None, metavar="FILE",
        help="保存しておいた「登る途中・登った後」の姿勢（g1_climb_collect_states.py で作成）から始めるエピソードを混ぜる",
    )
    parser.add_argument("--replay-prob", type=float, default=0.3, help="--replay の姿勢から始めるエピソードの割合")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    if args.resume is None:
        env_cfg, obs_cfg, reward_cfg, curriculum_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)
        if args.replay:
            env_cfg["reset"]["replay_file"] = os.path.abspath(args.replay)
            env_cfg["reset"]["replay_prob"] = args.replay_prob
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

    env = G1ClimbEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        curriculum_cfg=curriculum_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(os.path.join(log_dir, f"model_{args.resume}.pt"))
        # 補助の力のカリキュラムを、再開したイテレーションの位置から続ける
        env.common_step_counter = args.resume * train_cfg["num_steps_per_env"]

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
