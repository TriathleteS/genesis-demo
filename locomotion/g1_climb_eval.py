"""学習済みの G1 よじ登り方策を再生・評価するスクリプト。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_climb_eval.py                      # 最新のチェックポイントで、高さ 0.6 m の台に挑戦
    uv run python locomotion/g1_climb_eval.py -e g1-climb          # 1 回目の版（g1-climb）を再生する
    uv run python locomotion/g1_climb_eval.py --height 0.8         # 台の高さを変える
    uv run python locomotion/g1_climb_eval.py --ckpt 1500          # チェックポイントを指定する
    uv run python locomotion/g1_climb_eval.py --no-viewer --trials 20 --height 0.6
                                                                   # 画面なしで 20 回挑戦し、成功率を表示する
    uv run python locomotion/g1_climb_eval.py --record             # videos/ に録画する

1 回の挑戦は最大 10 秒。転倒するか 10 秒たつと、自動で台の手前に戻って次の挑戦を始める。

ビューアー上のキー操作:
    Enter     : 開始・一時停止（ビューアーは止まった状態で開くので、録画の準備ができたら押す）
    Backspace : 台の手前に戻して一時停止する（撮り直し用）
    R         : ビューアーの画面をそのまま録画する（Genesis の標準機能）
"""

import argparse
import glob
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

from g1_climb_env import G1ClimbEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-climb-v5", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("--ckpt", type=int, default=None, help="読み込むチェックポイントのイテレーション（省略時は最新）")
    parser.add_argument("--height", type=float, default=0.6, help="台の高さ [m]")
    parser.add_argument("--no-wait", action="store_true", help="Enter を待たずにすぐ始める")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーなしで --trials 回挑戦し、成功率を表示する")
    parser.add_argument("--trials", type=int, default=10, help="--no-viewer のときの挑戦回数")
    parser.add_argument(
        "--record", nargs="?", const="", default=None, metavar="FILE",
        help="mp4 に録画する（ファイル名を省略すると videos/g1_climb_<日時>.mp4）",
    )
    parser.add_argument("--record-res", type=str, default="1280x720", help="録画の解像度（幅x高さ）")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, curriculum_cfg, train_cfg = pickle.load(f)
    # 再生時は観測ノイズと補助の力を止め、ロボットの置き方のばらつきも小さくする
    obs_cfg["add_noise"] = False
    curriculum_cfg["assist_init"] = 0.0
    env_cfg["reset"]["yaw_range"] = 0.0
    env_cfg["reset"]["lateral_range"] = 0.0

    env = G1ClimbEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        curriculum_cfg=curriculum_cfg,
        show_viewer=not args.no_viewer,
        camera_res=tuple(int(v) for v in args.record_res.lower().split("x")) if args.record is not None else None,
    )
    env.set_box_height(args.height)
    print(f"台の高さ: {args.height:.2f} m")

    if args.ckpt is None:
        ckpts = [int(m.group(1)) for p in glob.glob(os.path.join(log_dir, "model_*.pt"))
                 if (m := re.search(r"model_(\d+)\.pt$", p))]
        if not ckpts:
            raise FileNotFoundError(f"{log_dir} にチェックポイント（model_*.pt）がありません。")
        args.ckpt = max(ckpts)
    print(f"チェックポイント model_{args.ckpt}.pt を読み込みます。")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # 開始・一時停止・やり直し
    state = {"running": args.no_viewer or args.no_wait, "reset": False}

    def toggle_run():
        state["running"] = not state["running"]
        print("開始しました。" if state["running"] else "一時停止しました（Enter で再開）。")

    def request_reset():
        state["reset"] = True

    if not args.no_viewer:
        env.scene.viewer.register_keybinds(
            Keybind("climb_start_pause", Key.ENTER, KeyAction.PRESS, callback=toggle_run),
            Keybind("climb_reset", Key.BACKSPACE, KeyAction.PRESS, callback=request_reset),
        )
        print("Enter: 開始・一時停止, Backspace: 台の手前に戻す")

    # 録画（動いている間だけ録画する）
    record_path = None
    if args.record is not None:
        record_path = os.path.abspath(
            args.record or os.path.join("videos", f"g1_climb_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        )
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
    rec = {"started": False, "active": False}

    def update_recording():
        if record_path is None or state["running"] == rec["active"]:
            return
        if state["running"]:
            if rec["started"]:
                env.camera.start_recording()
            else:
                env.camera.start_recording(save_to_filename=record_path, fps=1.0 / env.dt)
                print(f"録画を開始しました: {record_path}")
            rec.update(started=True, active=True)
        else:
            env.camera.pause_recording()
            rec["active"] = False

    # 挑戦ごとの結果
    results = []  # (成功したか, 登り切るまでの時間 [s], その後も台の上に立ち続けたか)
    trial = {"step": 0, "climb_time": None}

    def finish_trial(reason):
        ok = trial["climb_time"] is not None
        stayed = ok and reason == "時間切れ" and trial["on_top_at_end"]
        results.append((ok, trial["climb_time"], stayed))
        if ok:
            after = "時間切れまで台の上に立っていた" if stayed else "最後は台の上に立っていなかった"
            msg = f"{trial['climb_time']:.1f} 秒で台の上に立った（{after}）"
        else:
            msg = f"失敗（{reason}）"
        print(f"挑戦 {len(results)}: {msg}  最高到達度 {float(env.last_best_progress[0]):.2f}")

    obs_dict = env.reset()
    try:
        if not args.no_viewer:
            env.scene.viewer.update(force=True)
            if not state["running"]:
                print("準備ができました。ビューアーのウィンドウをクリックしてから Enter を押すと始まります。")
        with torch.no_grad():
            while not args.no_viewer or len(results) < args.trials:
                if state["reset"]:
                    obs_dict = env.reset()
                    trial.update(step=0, climb_time=None)
                    state.update(reset=False, running=False)
                    env.scene.viewer.update(force=True)
                    print("台の手前に戻しました。Enter で始まります。")
                update_recording()
                if not state["running"]:
                    env.scene.viewer.sleep(1 / 30)
                    continue
                actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)
                trial["step"] += 1
                if bool(env.last_just_succeeded[0]) and trial["climb_time"] is None:
                    trial["climb_time"] = trial["step"] * env.dt
                if bool(dones[0]):
                    trial["on_top_at_end"] = bool(env.last_on_top[0])
                    finish_trial("時間切れ" if bool(infos["time_outs"][0]) else "転倒")
                    trial.update(step=0, climb_time=None)
    except (gs.GenesisException, KeyboardInterrupt):
        pass
    finally:
        if rec["started"]:
            env.camera.stop_recording()
            print(f"録画を保存しました: {record_path}")

    if results:
        n_ok = sum(ok for ok, _, _ in results)
        n_stayed = sum(stayed for _, _, stayed in results)
        times = [t for ok, t, _ in results if ok]
        mean_time = f"、登り切るまでの平均 {sum(times) / len(times):.1f} 秒" if times else ""
        print(
            f"結果: 台の高さ {args.height:.2f} m で {len(results)} 回中 {n_ok} 回成功（{100 * n_ok / len(results):.0f}%）"
            f"{mean_time}。成功のうち {n_stayed} 回は時間切れまで台の上に立っていた"
        )


if __name__ == "__main__":
    main()
