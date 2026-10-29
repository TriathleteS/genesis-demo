"""学習済みのよじ登り方策を動かし、「登る途中・登った後」の姿勢を集めて保存するスクリプト。

集めた姿勢は、g1_climb_train.py の --replay オプションで、一部のエピソードの開始姿勢として使う。
（ゴールに近い状態から練習させることで、まだ一度も登り切れていない方策にも「登った後」や
「片足を台にかけた後」の経験をさせる。いわゆる逆向きのカリキュラム）

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_climb_collect_states.py -e g1-climb --ckpt 2999 --heights 0.5 0.6 0.7 0.8 0.9 1.0
    uv run python locomotion/g1_climb_collect_states.py -e g1-climb-v2 --ckpt 1500 --heights 0.2 --append

集める姿勢の条件（どれか 1 つを満たし、胴体の上下の速さが 1 m/s 未満で、体のどこかが床か台に触れている）:
    one_foot : 片足だけが台の上にある（もう一方の足を上げれば登れる状態）
    on_top   : 両足が台の上にあり、体を起こして立っている
    hands    : 手が台の上についている
"""

import argparse
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_climb_env import G1ClimbEnv

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climb_start_states.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, required=True, help="姿勢を集める方策の実験名")
    parser.add_argument("--ckpt", type=int, required=True, help="チェックポイントのイテレーション")
    parser.add_argument("--heights", type=float, nargs="+", required=True, help="台の高さ [m]（複数指定可）")
    parser.add_argument("--num-envs", type=int, default=32, help="並列に動かす環境の数")
    parser.add_argument("--duration", type=float, default=6.0, help="1 つの高さで動かす時間 [s]")
    parser.add_argument("--every", type=int, default=5, help="何ステップごとに姿勢を記録するか")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="保存先")
    parser.add_argument("--append", action="store_true", help="既存の保存先に追加する")
    parser.add_argument("--cpu", action="store_true", help="CPU で実行する")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, curriculum_cfg, train_cfg = pickle.load(f)
    obs_cfg["add_noise"] = False
    curriculum_cfg["assist_init"] = 0.0
    env_cfg["reset"]["replay_file"] = None

    env = G1ClimbEnv(args.num_envs, env_cfg, obs_cfg, reward_cfg, curriculum_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    all_qpos, all_height, all_kind = [], [], []
    for height in args.heights:
        env.set_box_height(height)
        obs = env.reset()
        counts = {"one_foot": 0, "on_top": 0, "hands": 0}
        with torch.no_grad():
            for step in range(int(args.duration / env.dt)):
                obs, _, dones, _ = env.step(policy(obs))
                if step % args.every:
                    continue
                calm = (torch.abs(env.base_vel_world[:, 2]) < 1.0) & env.in_contact & ~dones
                kinds = {
                    "one_foot": env.feet_on_box.sum(dim=1) == 1,
                    "on_top": env.on_top,
                    "hands": env.hands_on_box.any(dim=1),
                }
                for i, (name, cond) in enumerate(kinds.items()):
                    sel = calm & cond
                    if sel.any():
                        all_qpos.append(env.robot.get_qpos()[sel].cpu())
                        all_height.append(torch.full((int(sel.sum()),), height))
                        all_kind.append(torch.full((int(sel.sum()),), i))
                        counts[name] += int(sel.sum())
        print(f"台の高さ {height:.2f} m: 片足が台の上 {counts['one_foot']}, 台の上に立つ {counts['on_top']}, 手が台の上 {counts['hands']}")

    if not all_qpos:
        print("条件を満たす姿勢が 1 つも集まりませんでした。")
        return
    data = {"qpos": torch.cat(all_qpos), "height": torch.cat(all_height), "kind": torch.cat(all_kind)}
    if args.append and os.path.exists(args.out):
        old = torch.load(args.out)
        data = {k: torch.cat((old[k], data[k])) for k in data}
    torch.save(data, args.out)
    print(f"{len(data['qpos'])} 個の姿勢を保存しました: {args.out}")


if __name__ == "__main__":
    main()
