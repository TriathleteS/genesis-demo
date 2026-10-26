"""Franka アームを読み込み、PD 制御で複数の姿勢を順に巡回する最小サンプル。

実行方法（genesis-demo フォルダで）:
    uv run python examples/franka_pd_control.py          # GPU で実行
    uv run python examples/franka_pd_control.py --cpu    # CPU で実行
    uv run python examples/franka_pd_control.py --no-viewer  # ビューアーなしで実行
    uv run python examples/franka_pd_control.py --cycles 10  # 10 周させる（途中終了は Ctrl+C）
"""

import argparse

import numpy as np

import genesis as gs

# Franka の関節名（MJCF ファイル内の名前）。腕 7 関節 + グリッパー 2 関節 = 9 自由度
JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)

# PD 制御のゲインと出力上限（Genesis 公式チュートリアルの値）
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
FORCE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])  # 回転関節は N·m、指は N

# 巡回する目標姿勢 [rad]（指の 2 要素は開き幅 [m]、0.04 で全開・0.0 で閉）
HOME_POSE = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
TARGET_POSES = [
    np.array([0.8, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785, 0.04, 0.04]),  # 左へ旋回
    np.array([-0.8, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785, 0.0, 0.0]),  # 右へ旋回して指を閉じる
    np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.9, 0.785, 0.04, 0.04]),  # 前方へ伸ばす
    HOME_POSE,  # 初期姿勢へ戻る
]

MOVE_STEPS = 150  # 1 つの姿勢へ移動するのにかけるステップ数（dt=0.01 なので 1.5 秒）
HOLD_STEPS = 50  # 到着後に姿勢を保持するステップ数


def main() -> None:
    parser = argparse.ArgumentParser(description="Franka PD 制御サンプル")
    parser.add_argument("--cpu", action="store_true", help="CPU バックエンドで実行する")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーを表示しない")
    parser.add_argument("--cycles", type=int, default=2, help="姿勢の巡回を何周するか（既定: 2）")
    args = parser.parse_args()

    # 1. Genesis の初期化（必ず最初に呼ぶ）
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    # 2. シーンの作成
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.0, 1.8),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=not args.no_viewer,
    )

    # 3. 地面と Franka アームを追加（Franka のモデルは Genesis 同梱のものを使用）
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    # 4. シーンをビルド（ここで物理エンジンのカーネルがコンパイルされる。初回は時間がかかる）
    scene.build()

    # 5. 関節名 → 自由度インデックスの対応を取得
    dofs_idx = [franka.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]

    # 6. PD ゲインと出力上限を設定
    franka.set_dofs_kp(kp=KP, dofs_idx_local=dofs_idx)
    franka.set_dofs_kv(kv=KV, dofs_idx_local=dofs_idx)
    franka.set_dofs_force_range(lower=-FORCE_LIMIT, upper=FORCE_LIMIT, dofs_idx_local=dofs_idx)

    # 7. 初期姿勢へ直接セット（set_* は物理計算を経ずに状態を書き換える。リセット用）
    franka.set_dofs_position(HOME_POSE, dofs_idx)
    franka.control_dofs_position(HOME_POSE, dofs_idx)  # 重力で落ちないよう保持目標も設定

    # 8. 制御ループ：目標姿勢へ滑らかに補間しながら PD 制御で追従させる
    current = HOME_POSE.copy()
    for cycle in range(args.cycles):
        for pose_no, target in enumerate(TARGET_POSES, start=1):
            start = current
            for step in range(MOVE_STEPS + HOLD_STEPS):
                # 0→1 へ滑らかに変化する補間係数（加減速のある S 字カーブ）
                s = min(step / MOVE_STEPS, 1.0)
                alpha = 0.5 - 0.5 * np.cos(np.pi * s)
                franka.control_dofs_position(start + alpha * (target - start), dofs_idx)
                scene.step()

            current = target
            # 実際の関節角度と目標との誤差を表示（腕の 7 関節のみ）
            actual = franka.get_dofs_position(dofs_idx).cpu().numpy()
            err = np.abs(actual[:7] - target[:7]).max()
            print(f"[周回 {cycle + 1}] 姿勢 {pose_no} に到達  最大関節誤差: {np.degrees(err):.2f} 度")

    print("完了しました。")


if __name__ == "__main__":
    main()
