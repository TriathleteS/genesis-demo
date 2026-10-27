"""学習済みの G1 歩行方策をビューアーで再生するスクリプト。キーボードで速度指令を変えられる。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_eval.py --ckpt 1000              # logs/g1-walking/model_1000.pt を再生
    uv run python locomotion/g1_eval.py --ckpt 1000 --vx 0.8     # 最初の指令を前進 0.8 m/s にする

ビューアー上のキー操作:
    ↑ / ↓ : 前進・後退の速度を 0.1 m/s ずつ変える
    ← / → : 旋回の速度を 0.1 rad/s ずつ変える（← が左旋回）
    , / . : 左右の速度を 0.1 m/s ずつ変える（, が左）
    Space : 停止（全ての指令を 0 にする）
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
from genesis.vis.keybindings import Key, KeyAction, Keybind

from g1_env import G1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-walking", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("--ckpt", type=int, default=1000, help="読み込むチェックポイントのイテレーション")
    parser.add_argument("--vx", type=float, default=0.5, help="最初の前後速度の指令 [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="最初の左右速度の指令 [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="最初の旋回速度の指令 [rad/s]")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーなしで --duration 秒だけ動かし、結果を表示する")
    parser.add_argument("--duration", type=float, default=10.0, help="--no-viewer のときに動かす時間 [s]")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    # 再生時は報酬の計算・外乱・観測ノイズ・摩擦のランダム化を止める
    reward_cfg["reward_scales"] = {}
    env_cfg["push_robots"] = False
    env_cfg["randomize_friction"] = False
    obs_cfg["add_noise"] = False

    env = G1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_viewer,
    )
    env.max_episode_length = 10**9  # 学習時の 20 秒の時間切れで初期位置に戻らないようにする

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # 速度指令（学習した範囲に収める）
    cmd = {"vx": args.vx, "vy": args.vy, "wz": args.wz}
    limits = {
        "vx": command_cfg["lin_vel_x_range"],
        "vy": command_cfg["lin_vel_y_range"],
        "wz": command_cfg["ang_vel_range"],
    }

    def apply_command():
        for key, (lo, hi) in limits.items():
            cmd[key] = round(min(max(cmd[key], lo), hi), 2)
        env.set_command(cmd["vx"], cmd["vy"], cmd["wz"])
        print(f"速度指令: 前後 {cmd['vx']:+.1f} m/s, 左右 {cmd['vy']:+.1f} m/s, 旋回 {cmd['wz']:+.1f} rad/s")

    def change(key, delta):
        cmd[key] += delta
        apply_command()

    def stop():
        cmd.update(vx=0.0, vy=0.0, wz=0.0)
        apply_command()

    apply_command()

    if not args.no_viewer:
        env.scene.viewer.follow_entity(env.robot, smoothing=0.9)
        env.scene.viewer.register_keybinds(
            Keybind("g1_forward", Key.UP, KeyAction.PRESS, callback=change, args=("vx", 0.1)),
            Keybind("g1_backward", Key.DOWN, KeyAction.PRESS, callback=change, args=("vx", -0.1)),
            Keybind("g1_turn_left", Key.LEFT, KeyAction.PRESS, callback=change, args=("wz", 0.1)),
            Keybind("g1_turn_right", Key.RIGHT, KeyAction.PRESS, callback=change, args=("wz", -0.1)),
            Keybind("g1_step_left", Key.COMMA, KeyAction.PRESS, callback=change, args=("vy", 0.1)),
            Keybind("g1_step_right", Key.PERIOD, KeyAction.PRESS, callback=change, args=("vy", -0.1)),
            Keybind("g1_stop", Key.SPACE, KeyAction.PRESS, callback=stop),
        )
        print("↑↓: 前後, ←→: 旋回, , .: 左右, Space: 停止")

    obs_dict = env.reset()
    n_steps = int(args.duration / env.dt) if args.no_viewer else None
    n_falls, step = 0, 0
    try:
        with torch.no_grad():
            while n_steps is None or step < n_steps:
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
                n_falls += int(dones[0]) - int(infos["time_outs"][0])
                step += 1
    except (gs.GenesisException, KeyboardInterrupt):
        # ビューアーを閉じると次のステップで例外が出るので、それを終了の合図として扱う
        pass

    pos = env.robot.get_pos()[0].tolist()
    print(f"再生を終了しました。最終位置 x={pos[0]:+.2f} m, y={pos[1]:+.2f} m, 高さ={pos[2]:.2f} m, 転倒回数 {n_falls} 回")


if __name__ == "__main__":
    main()
