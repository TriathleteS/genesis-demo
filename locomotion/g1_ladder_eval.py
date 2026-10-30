"""学習済みの G1 はしご登り方策を再生・評価するスクリプト。

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_ladder_eval.py                     # 最新のチェックポイントで、傾き 85° のはしごに挑戦
    uv run python locomotion/g1_ladder_eval.py --angle 90          # はしごの傾きを変える（学習は 80〜90°）
    uv run python locomotion/g1_ladder_eval.py --ckpt 1500         # チェックポイントを指定する
    uv run python locomotion/g1_ladder_eval.py --no-viewer --trials 20 --angle 75
                                                                   # 画面なしで 20 回挑戦し、成功率を表示する
    uv run python locomotion/g1_ladder_eval.py --record            # videos/ に録画する
    uv run python locomotion/g1_ladder_eval.py --start hang --angle 90
                                                                   # はしごの高い段を両手で握ってぶら下がった姿勢から始める
    uv run python locomotion/g1_ladder_eval.py --start hang --hang-rung 6
                                                                   # ぶら下がるときに握る段を指定する（0 が一番下。学習では 5〜7、v11a からは 5〜10）
    uv run python locomotion/g1_ladder_eval.py --start stand --angle 90
                                                                   # 床に立ったまま、肩より少し上の段を両手で握った姿勢から始める

1 回の挑戦は最大 15 秒。転倒するか、15 秒たつか、4 秒間進まなかった（停滞）と、自動ではしごの手前に戻って次の挑戦を始める。
成功：両足が目標の段（v10 までは下から 3 段目、v11a からは 6 段目）以上の横棒に乗り、少なくとも片手で横棒を握っている。

ビューアー上のキー操作:
    Enter     : 開始・一時停止（ビューアーは止まった状態で開くので、録画の準備ができたら押す）
    Backspace : 最初の姿勢に戻して一時停止する（撮り直し用）
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

from g1_ladder_env import G1LadderEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="g1-ladder-v11a", help="実験名（logs/ の下のフォルダ名）")
    parser.add_argument("--ckpt", type=int, default=None, help="読み込むチェックポイントのイテレーション（省略時は最新）")
    parser.add_argument("--angle", type=float, default=85.0, help="はしごの傾き（床からの角度）[deg]")
    parser.add_argument(
        "--start", choices=["ground", "hang", "stand"], default="ground",
        help="開始の仕方（ground：床から、hang：ぶら下がった姿勢から、stand：床に立って握った姿勢から）",
    )
    parser.add_argument("--hang-rung", type=int, default=None, help="--start hang で握る段（0 が一番下。省略時は 5〜7 からランダム）")
    parser.add_argument("--no-wait", action="store_true", help="Enter を待たずにすぐ始める")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーなしで --trials 回挑戦し、成功率を表示する")
    parser.add_argument("--trials", type=int, default=10, help="--no-viewer のときの挑戦回数")
    parser.add_argument(
        "--record", nargs="?", const="", default=None, metavar="FILE",
        help="mp4 に録画する（ファイル名を省略すると videos/g1_ladder_<日時>.mp4）",
    )
    parser.add_argument("--record-res", type=str, default="1280x720", help="録画の解像度（幅x高さ）")
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if not os.path.exists(f"{log_dir}/cfgs.pkl"):
        found = sorted(d for d in os.listdir("logs") if d.startswith("g1-ladder")) if os.path.isdir("logs") else []
        raise SystemExit(
            f"{log_dir} に学習結果がありません。-e で実験名を指定してください"
            f"（logs にあるはしごの実験：{', '.join(found) or 'なし'}）。"
        )
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, curriculum_cfg, train_cfg = pickle.load(f)
    # 再生時は観測ノイズと補助の力を止め、ロボットの置き方のばらつきも小さくする
    obs_cfg["add_noise"] = False
    curriculum_cfg["assist_init"] = 0.0
    env_cfg["reset"]["yaw_range"] = 0.0
    env_cfg["reset"]["lateral_range"] = 0.0

    env = G1LadderEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        curriculum_cfg=curriculum_cfg,
        show_viewer=not args.no_viewer,
        camera_res=tuple(int(v) for v in args.record_res.lower().split("x")) if args.record is not None else None,
        rail_angle_deg=args.angle,
    )
    env.set_eval_start(args.start, args.hang_rung)
    env.set_ladder_angle(args.angle)
    start_name = {"ground": "床から", "hang": "ぶら下がった姿勢から", "stand": "床に立って握った姿勢から"}[args.start]
    print(f"はしごの傾き: {args.angle:.1f}°、開始: {start_name}")

    if args.ckpt is None:
        ckpts = [int(m.group(1)) for p in glob.glob(os.path.join(log_dir, "model_*.pt"))
                 if (m := re.search(r"model_(\d+)\.pt$", p))]
        if not ckpts:
            raise FileNotFoundError(f"{log_dir} にチェックポイント（model_*.pt）がありません。")
        args.ckpt = max(ckpts)
    print(f"チェックポイント model_{args.ckpt}.pt を読み込みます。")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"), map_location=gs.device)
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
        print("Enter: 開始・一時停止, Backspace: 最初の姿勢に戻す")

    # 録画（動いている間だけ録画する）
    record_path = None
    if args.record is not None:
        record_path = os.path.abspath(
            args.record or os.path.join("videos", f"g1_ladder_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
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
    results = []  # (成功したか, 登り切るまでの時間 [s], その後も目標の段に留まり続けたか)
    trial = {"step": 0, "climb_time": None}

    def finish_trial(reason):
        ok = trial["climb_time"] is not None
        stayed = ok and reason == "時間切れ" and trial["on_top_at_end"]
        results.append((ok, trial["climb_time"], stayed))
        if ok:
            after = "時間切れまで目標の段に留まった" if stayed else "最後は目標の段にいなかった"
            msg = f"{trial['climb_time']:.1f} 秒で目標の段に到達（{after}）"
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
                    print("最初の姿勢に戻しました。Enter で始まります。")
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
                    trial["on_top_at_end"] = bool(env.last_on_target[0])
                    if bool(infos["time_outs"][0]):
                        reason = "時間切れ"
                    elif bool(getattr(env, "last_stalled", torch.zeros(1, dtype=torch.bool))[0]):
                        reason = "停滞：4 秒間進まなかった"
                    else:
                        reason = "転倒"
                    finish_trial(reason)
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
            f"結果: 傾き {args.angle:.1f}° のはしごで {len(results)} 回中 {n_ok} 回成功（{100 * n_ok / len(results):.0f}%）"
            f"{mean_time}。成功のうち {n_stayed} 回は時間切れまで目標の段に留まった"
        )


if __name__ == "__main__":
    main()
