"""学習済みの G1 歩行方策をビューアーで再生するスクリプト。キーボードで速度指令を変えられる。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_eval.py                 # logs/g1-walking/ の最新のチェックポイントを再生
    uv run python locomotion/g1_eval.py --ckpt 500      # logs/g1-walking/model_500.pt を再生
    uv run python locomotion/g1_eval.py --vx 0.8        # 最初の指令を前進 0.8 m/s にする

1000 イテレーション学習した場合、最後のチェックポイントは model_999.pt（0 から数えるため）。

ビューアー上のキー操作:
    ↑ / ↓ : 前進・後退の速度を 0.1 m/s ずつ変える
    ← / → : 旋回の速度を 0.1 rad/s ずつ変える（← が左旋回）
    , / . : 左右の速度を 0.1 m/s ずつ変える（, が左）
    Space : 停止（全ての指令を 0 にする）
    Enter : 歩行の開始・一時停止（ビューアーは止まった状態で開くので、録画の準備ができたら押す）
    Backspace : 初期位置に戻して一時停止する（撮り直し用）
    R : ビューアーの画面をそのまま録画する（もう一度 R で停止し、保存先を選ぶ。Genesis の標準機能）

動画ファイルに録画する場合:
    uv run python locomotion/g1_eval.py --record                  # videos/ に日時入りのファイル名で保存
    uv run python locomotion/g1_eval.py --record walk.mp4         # ファイル名を指定して保存
    uv run python locomotion/g1_eval.py --record --no-viewer --duration 20   # ビューアーなしで 20 秒撮る
  ロボットを追いかけるカメラで、歩いている間（Enter で開始してから一時停止・終了まで）だけを録画する。
  ビューアーの見た目やウィンドウの大きさとは関係なく、--record-res の解像度（既定 1280x720）で保存される。

旋回の指令が 0 のあいだは、歩き始めた向き（または最後に旋回し終えた向き）を保つよう、
旋回速度の指令を自動で補正する（--no-heading-hold で無効化）。
"""

import argparse
import glob
import math
import os
import pickle
import re
import time
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
    parser.add_argument("--ckpt", type=int, default=None, help="読み込むチェックポイントのイテレーション（省略時は最新）")
    parser.add_argument("--vx", type=float, default=0.5, help="最初の前後速度の指令 [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="最初の左右速度の指令 [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="最初の旋回速度の指令 [rad/s]")
    parser.add_argument("--no-heading-hold", action="store_true", help="向きを保つ自動補正を無効にする")
    parser.add_argument("--no-wait", action="store_true", help="Enter を待たずにすぐ歩き始める")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーなしで --duration 秒だけ動かし、結果を表示する")
    parser.add_argument("--duration", type=float, default=10.0, help="--no-viewer のときに動かす時間 [s]")
    parser.add_argument(
        "--record", nargs="?", const="", default=None, metavar="FILE",
        help="歩行を mp4 に録画する（ファイル名を省略すると videos/g1_<日時>.mp4）",
    )
    parser.add_argument("--record-res", type=str, default="1280x720", help="録画の解像度（幅x高さ）")
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
        camera_res=tuple(int(v) for v in args.record_res.lower().split("x")) if args.record is not None else None,
    )
    env.max_episode_length = 10**9  # 学習時の 20 秒の時間切れで初期位置に戻らないようにする

    if args.ckpt is None:
        # model_<番号>.pt のうち、番号が最大のものを選ぶ
        ckpts = [int(m.group(1)) for p in glob.glob(os.path.join(log_dir, "model_*.pt"))
                 if (m := re.search(r"model_(\d+)\.pt$", p))]
        if not ckpts:
            raise FileNotFoundError(f"{log_dir} にチェックポイント（model_*.pt）がありません。")
        args.ckpt = max(ckpts)
    print(f"チェックポイント model_{args.ckpt}.pt を読み込みます。")

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

    # 方策は「旋回速度」の指令に従うだけで、向きそのものは見ていない。そのため旋回速度 0 の指令でも
    # わずかな誤差が積み重なって少しずつ曲がっていく。旋回の指令が 0 のあいだは、目標の向きとのずれに
    # 比例した旋回速度を指令して、まっすぐ歩かせる（Unitree の学習コードの heading_command と同じ考え方）
    heading = {"target": None}

    def effective_wz():
        yaw = float(env.base_euler[0, 2])
        if args.no_heading_hold or cmd["wz"] != 0.0 or heading["target"] is None:
            heading["target"] = yaw  # 旋回中は、今向いている方向を次の目標にする
            return cmd["wz"]
        error = (heading["target"] - yaw + math.pi) % (2 * math.pi) - math.pi  # -π〜π に収める
        lo, hi = limits["wz"]
        return min(max(1.0 * error, lo), hi)

    # 再生の開始・一時停止・やり直し（ビューアーのキー操作から切り替える）
    state = {"running": args.no_viewer or args.no_wait, "reset": False}

    def toggle_run():
        state["running"] = not state["running"]
        print("歩行を開始しました。" if state["running"] else "一時停止しました（Enter で再開）。")

    def request_reset():
        state["reset"] = True

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
            Keybind("g1_start_pause", Key.ENTER, KeyAction.PRESS, callback=toggle_run),
            Keybind("g1_reset", Key.BACKSPACE, KeyAction.PRESS, callback=request_reset),
        )
        print("↑↓: 前後, ←→: 旋回, , .: 左右, Space: 停止, Enter: 開始・一時停止, Backspace: 初期位置に戻す")

    # 録画の準備（歩いている間だけ録画し、一時停止中は録画も止める）
    record_path = None
    if args.record is not None:
        record_path = os.path.abspath(args.record or os.path.join("videos", f"g1_{time.strftime('%Y%m%d_%H%M%S')}.mp4"))
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        env.camera.follow_entity(env.robot, smoothing=0.9)
    rec = {"started": False, "active": False}

    def update_recording():
        if record_path is None or state["running"] == rec["active"]:
            return
        if state["running"]:
            if rec["started"]:
                env.camera.start_recording()  # 一時停止していた録画を再開する
            else:
                env.camera.start_recording(save_to_filename=record_path, fps=1.0 / env.dt)
                print(f"録画を開始しました: {record_path}")
            rec.update(started=True, active=True)
        else:
            env.camera.pause_recording()
            rec["active"] = False

    obs_dict = env.reset()
    n_steps = int(args.duration / env.dt) if args.no_viewer else None
    n_falls, step = 0, 0
    try:
        if not args.no_viewer:
            env.scene.viewer.update(force=True)  # 歩き始める前の立ち姿勢を表示する
            if not state["running"]:
                print("準備ができました。ビューアーのウィンドウをクリックしてから Enter を押すと歩き始めます。")
        with torch.no_grad():
            while n_steps is None or step < n_steps:
                if state["reset"]:
                    obs_dict = env.reset()
                    env.base_euler.zero_()  # 向きの自動補正が、リセット前の向きを目標にしないようにする
                    heading["target"] = None
                    n_falls, step = 0, 0
                    state.update(reset=False, running=False)
                    env.scene.viewer.update(force=True)
                    print("初期位置に戻しました。Enter で歩き始めます。")
                update_recording()
                if not state["running"]:
                    # シミュレーションを進めずに、ウィンドウの描画とキー操作の受け付けだけを続ける
                    env.scene.viewer.sleep(1 / 30)
                    continue
                env.set_command(cmd["vx"], cmd["vy"], effective_wz())
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
                n_falls += int(dones[0]) - int(infos["time_outs"][0])
                step += 1
    except (gs.GenesisException, KeyboardInterrupt):
        # ビューアーを閉じると次のステップで例外が出るので、それを終了の合図として扱う
        pass
    finally:
        if rec["started"]:
            env.camera.stop_recording()
            print(f"録画を保存しました: {record_path}")

    pos = env.robot.get_pos()[0].tolist()
    elapsed = step * env.dt
    speed = math.hypot(pos[0], pos[1]) / elapsed if elapsed > 0 else 0.0
    print(
        f"再生を終了しました。{elapsed:.1f} 秒間、最終位置 x={pos[0]:+.2f} m, y={pos[1]:+.2f} m, 高さ={pos[2]:.2f} m, "
        f"向き {math.degrees(float(env.base_euler[0, 2])):+.0f} 度, 平均速度 {speed:.2f} m/s, 転倒回数 {n_falls} 回"
    )


if __name__ == "__main__":
    main()
