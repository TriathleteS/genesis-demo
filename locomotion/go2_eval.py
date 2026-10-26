"""学習済みの Go2 歩行方策をビューアーで再生するスクリプト。

Genesis World v1.4.5 の公式サンプル examples/locomotion/go2_eval.py を取り込み、
ビューアーを閉じたときや Ctrl+C で正常終了するようにしたもの（Apache License 2.0）。
https://github.com/Genesis-Embodied-AI/genesis-world/tree/v1.4.5/examples/locomotion

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/go2_eval.py --ckpt 100   # logs/go2-walking/model_100.pt を再生
"""

import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
    except (gs.GenesisException, KeyboardInterrupt):
        # ビューアーを閉じると次のステップで例外が出るので、それを終了の合図として扱う
        print("再生を終了しました。")


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
