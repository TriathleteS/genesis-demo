"""4 足歩行ロボット Unitree Go2 をトロット歩容で前進させるサンプル。

学習済みの方策（強化学習）は使わず、足先の軌道を数式で作り、それを逆運動学で
関節角度に変換して PD 制御で追従させる「オープンループ歩行」です。
姿勢のフィードバックが無いので、歩幅や周期を大きくしすぎると転倒します。

実行方法（genesis-demo フォルダで）:
    uv run python examples/go2_trot.py                    # 既定の設定で 20 秒歩く
    uv run python examples/go2_trot.py --duration 60      # 60 秒歩く（途中終了は Ctrl+C）
    uv run python examples/go2_trot.py --step-length 0.16 # 歩幅を広げる
    uv run python examples/go2_trot.py --cpu              # CPU で実行
"""

import argparse

import numpy as np

import genesis as gs

# ---------------------------------------------------------------------------
# ロボットの定数
# ---------------------------------------------------------------------------

# 脚の並び順（FL: 左前, FR: 右前, RL: 左後, RR: 右後）
LEGS = ("FL", "FR", "RL", "RR")

# 各脚の関節（hip: 左右に開く, thigh: 太もも, calf: すね）
JOINT_NAMES = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]

# 太もも・すねの長さ [m]（go2.urdf より）
THIGH_LEN = 0.213
CALF_LEN = 0.213

# トロット歩容：対角の脚（左前と右後、右前と左後）が同時に動く。値は歩行周期内の位相のずれ
GAIT_PHASE_OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}

# PD ゲインと出力上限（出力上限は go2.urdf の effort 上限に合わせる）
# ゲインが低いと脚が目標軌道に追従しきれず、遊脚が地面を擦って後ろ向きに進んでしまう
KP = 150.0
KV = 4.0
TORQUE_LIMIT = np.array([23.7, 23.7, 35.55] * 4)  # [N·m]（hip, thigh, calf の順 × 4 脚）

# 立った状態での足先の位置（各脚の股関節から見た位置）[m]
STAND_FOOT_X = 0.0  # 前後方向（+ が前）
STAND_FOOT_Z = -0.30  # 上下方向（- が下）

SPAWN_HEIGHT = 0.36  # 胴体の初期高さ [m]
DT = 0.005  # シミュレーションの時間刻み [s]


# ---------------------------------------------------------------------------
# 脚の逆運動学と足先軌道
# ---------------------------------------------------------------------------


def leg_ik(x: float, z: float) -> tuple[float, float]:
    """股関節から見た足先位置 (x, z) から、thigh・calf の関節角度 [rad] を求める（2 リンクの逆運動学）。

    Go2 の thigh・calf は y 軸まわりの回転で、角度が正のとき足先は後ろへ振れる。
    calf は常に負（膝が後ろ向きに曲がる）。
    """
    u, w = -x, -z  # 下向き・後ろ向きを正にした座標
    dist = np.clip(np.hypot(u, w), 1e-6, THIGH_LEN + CALF_LEN - 1e-6)  # 届かない位置は伸び切りで打ち切る
    u, w = u * dist / np.hypot(u, w), w * dist / np.hypot(u, w)

    cos_calf = (dist**2 - THIGH_LEN**2 - CALF_LEN**2) / (2 * THIGH_LEN * CALF_LEN)
    calf = -np.arccos(np.clip(cos_calf, -1.0, 1.0))
    thigh = np.arctan2(u, w) - np.arctan2(CALF_LEN * np.sin(calf), THIGH_LEN + CALF_LEN * np.cos(calf))
    return thigh, calf


def foot_trajectory(phase: float, step_length: float, step_height: float) -> tuple[float, float]:
    """歩行周期内の位相 phase（0〜1）における足先位置 (x, z) を返す。

    前半 (0〜0.5) は接地期：足先は地面上を前から後ろへ等速で動き、胴体を前に押し出す。
    後半 (0.5〜1) は遊脚期：足先を持ち上げながら後ろから前へ戻す。
    """
    if phase < 0.5:
        s = phase / 0.5
        x = step_length / 2 - step_length * s
        z = STAND_FOOT_Z
    else:
        s = (phase - 0.5) / 0.5
        x = -step_length / 2 + step_length * (0.5 - 0.5 * np.cos(np.pi * s))  # 加減速付きで前へ戻す
        z = STAND_FOOT_Z + step_height * np.sin(np.pi * s)  # 弧を描いて持ち上げる
    return STAND_FOOT_X + x, z


def compute_joint_targets(t: float, period: float, step_length: float, step_height: float) -> np.ndarray:
    """時刻 t における 12 関節の目標角度を、JOINT_NAMES の順で返す。"""
    targets = []
    for leg in LEGS:
        phase = (t / period + GAIT_PHASE_OFFSET[leg]) % 1.0
        x, z = foot_trajectory(phase, step_length, step_height)
        thigh, calf = leg_ik(x, z)
        targets += [0.0, thigh, calf]  # hip は 0（脚を開かない）
    return np.array(targets)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 トロット歩行サンプル")
    parser.add_argument("--cpu", action="store_true", help="CPU バックエンドで実行する")
    parser.add_argument("--no-viewer", action="store_true", help="ビューアーを表示しない")
    parser.add_argument("--duration", type=float, default=20.0, help="歩く時間 [s]（既定: 20）")
    parser.add_argument("--period", type=float, default=0.4, help="1 歩行周期の時間 [s]（既定: 0.4）")
    parser.add_argument("--step-length", type=float, default=0.12, help="歩幅 [m]（既定: 0.12）")
    parser.add_argument("--step-height", type=float, default=0.06, help="足の持ち上げ高さ [m]（既定: 0.06）")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.6, 0.8),  # follow_entity 使用時はロボットからの相対位置になる
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=DT),
        show_viewer=not args.no_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    go2 = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",  # Genesis 同梱のモデル
            pos=(0.0, 0.0, SPAWN_HEIGHT),
        ),
    )

    # カメラがロボットを追いかけるようにする（smoothing でカメラの揺れを抑える）
    if not args.no_viewer:
        scene.viewer.follow_entity(go2, smoothing=0.9)

    scene.build()

    dofs_idx = [go2.get_joint(name).dofs_idx_local[0] for name in JOINT_NAMES]
    go2.set_dofs_kp(kp=np.full(12, KP), dofs_idx_local=dofs_idx)
    go2.set_dofs_kv(kv=np.full(12, KV), dofs_idx_local=dofs_idx)
    go2.set_dofs_force_range(lower=-TORQUE_LIMIT, upper=TORQUE_LIMIT, dofs_idx_local=dofs_idx)

    # 1. 立ち姿勢をセットし、1 秒かけて地面に落ち着かせる
    stand = compute_joint_targets(0.0, args.period, step_length=0.0, step_height=0.0)
    go2.set_dofs_position(stand, dofs_idx)
    go2.control_dofs_position(stand, dofs_idx)
    for _ in range(int(1.0 / DT)):
        scene.step()
    print("立ち姿勢になりました。歩き始めます。")

    # 2. 歩行ループ（最初の 2 秒で歩幅を 0 から徐々に広げて、急発進で転ばないようにする）
    ramp_time = 2.0
    n_steps = int(args.duration / DT)
    for i in range(n_steps):
        t = i * DT
        ramp = min(t / ramp_time, 1.0)
        targets = compute_joint_targets(t, args.period, args.step_length * ramp, args.step_height * ramp)
        go2.control_dofs_position(targets, dofs_idx)
        scene.step()

        # 1 秒ごとに胴体の位置を表示し、転倒していないか確認する
        if i % int(1.0 / DT) == 0:
            pos = go2.get_pos().cpu().numpy()
            print(f"t={t:5.1f}s  胴体位置 x={pos[0]:+.2f} m, y={pos[1]:+.2f} m, 高さ={pos[2]:.2f} m")
            if pos[2] < 0.15:
                print("胴体が低すぎます。転倒したため終了します。歩幅や周期を小さくして試してください。")
                break

    pos = go2.get_pos().cpu().numpy()
    print(f"完了しました。前進距離: {pos[0]:.2f} m")


if __name__ == "__main__":
    main()
