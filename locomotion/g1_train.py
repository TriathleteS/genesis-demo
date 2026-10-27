"""ヒューマノイド Unitree G1 の歩行方策を強化学習（PPO）で学習するスクリプト。

Go2 の公式サンプル go2_train.py と同じ構成で、学習の設定は Unitree 公式の
unitree_rl_gym（BSD 3-Clause License）の G1 の設定を移植したもの。
https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_train.py                          # 4096 環境 × 1000 イテレーション
    uv run python locomotion/g1_train.py --max-iterations 3000    # もっと長く学習する
    uv run python locomotion/g1_train.py --resume 999 --max-iterations 1000
                                                                  # model_999.pt から続けて 1000 イテレーション学習する

学習結果は logs/<実験名>/ に保存される（model_<イテレーション>.pt が方策のチェックポイント）。
イテレーションは 0 から数えるので、100 回ごとの model_0.pt, model_100.pt, ... と、最後の model_999.pt（1000 回学習した場合）が保存される。
--resume を付けずに同じ実験名で実行すると、前回の logs/<実験名>/ は削除されて上書きされる点に注意。
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

from g1_env import G1Env


def get_train_cfg(exp_name):
    train_cfg_dict = {
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
        # Unitree は小さな LSTM を使っているが、ここでは Go2 と同じ MLP にしている（位相を観測に含めるので記憶は不要）
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,  # 観測を自動で正規化して学習を安定させる
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["critic"],  # 価値関数には胴体の速度を含む観測を与える
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    env_cfg = {
        "num_actions": 12,
        "joint_names": joint_names,
        # 行動が 0 のときの目標姿勢 [rad]（膝を軽く曲げた立ち姿勢）
        "default_joint_angles": {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
        },
        "feet_link_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "termination_contact_link_names": ["pelvis"],  # 胴体（腕・頭も一体化されている）が床に触れたら転倒
        "hip_joint_names": ["left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_yaw_joint"],
        # PD 制御 [N·m/rad]（関節名の部分一致で指定）
        "stiffness": {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40},
        "damping": {"hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2},
        "action_scale": 0.25,
        "clip_actions": 100.0,
        # シミュレーション
        "dt": 0.02,
        "substeps": 4,
        "max_collision_pairs": 40,
        # 初期姿勢
        "base_init_pos": [0.0, 0.0, 0.8],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # エピソード・終了条件
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,
        "termination_if_roll_greater_than": 0.8,  # [rad]
        "termination_if_pitch_greater_than": 1.0,  # [rad]
        # 歩行の位相
        "gait_period": 0.8,  # 1 歩行周期 [s]
        "stance_ratio": 0.55,  # 周期のうち足を接地させておく割合
        # ドメインランダム化
        "randomize_friction": True,
        "friction_range": [0.1, 1.25],
        "push_robots": True,
        "push_interval_s": 5.0,
        "max_push_vel_xy": 1.5,  # [m/s]
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales": {
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "ang_vel": 0.2,
            "gravity": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.78,
        "feet_swing_height_target": 0.08,
        "soft_dof_pos_limit": 0.9,
        "only_positive_rewards": True,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -10.0,
            "dof_acc": -2.5e-7,
            "dof_vel": -1e-3,
            "action_rate": -0.01,
            "dof_pos_limits": -5.0,
            "alive": 0.15,
            "hip_pos": -1.0,
            "contact_no_vel": -0.2,
            "feet_swing_height": -20.0,
            "contact": 0.18,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 1.0],  # 前後 [m/s]
        "lin_vel_y_range": [-0.3, 0.3],  # 左右 [m/s]
        "ang_vel_range": [-0.5, 0.5],  # 旋回 [rad/s]
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-walking", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="並列に動かす環境の数")
    parser.add_argument("--max-iterations", type=int, default=1000, help="学習のイテレーション数")
    parser.add_argument("--resume", type=int, default=None, help="このイテレーションのチェックポイントから学習を再開する")
    parser.add_argument("--seed", type=int, default=1, help="乱数のシード")
    parser.add_argument("--cpu", action="store_true", help="CPU で実行する（動作確認用。学習には GPU が必要）")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"

    if args.resume is None:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    else:
        # 再開時は、前回の学習と同じ設定を使う
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env = G1Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(os.path.join(log_dir, f"model_{args.resume}.pt"))

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
