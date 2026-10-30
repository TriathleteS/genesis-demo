"""はしごにつかまった姿勢を G1 に取らせ、その姿勢を保つのに各関節が出す必要のあるトルクを測るスクリプト。

強化学習とは別の確認用。「片足で 1 段目に立ち、もう一方の足を 2 段目へ上げる」のような姿勢が、
関節の最大トルクと握力の範囲内で保てるか（物理的に可能か）を調べる。

手順:
  1. 腰の位置・手足を置く段を決め、逆運動学（IK）で関節角度を求める（腰は固定し、23 関節だけを動かす）
  2. その姿勢でロボットを置き、手のひらを横棒にばねで引きつけ（はしご登りの学習と同じ「握る」動作）、
     関節を PD 制御でその角度に保ったまま settle_time 秒待つ（腰は固定しない。重力で崩れるなら崩れる）
  3. 最後の 0.5 秒の平均で、各関節のトルク（最大値に対する割合）・握る力・足にかかる力・姿勢の崩れを表示する

実行方法（genesis-demo フォルダで）:
    uv run python locomotion/g1_ladder_pose_check.py                     # 用意した姿勢をすべて、90° と 75° で調べる
    uv run python locomotion/g1_ladder_pose_check.py --angles 90 --poses step_up
    uv run python locomotion/g1_ladder_pose_check.py --angles 90 --poses step_up --viewer   # 姿勢を画面で見る
    uv run python locomotion/g1_ladder_pose_check.py --stiffness-scale 3                    # 関節の PD のばねを強くして測る
    uv run python locomotion/g1_ladder_pose_check.py --stiffness-scale 5 --urdf g1_23dof_rev_1_0_sole.urdf \
        --angles 70 75 80 85 90 --poses both_on_0 step_up --base-n 0.2 0.25 0.3 0.35 0.4 0.45 --summary
                                                                   # 傾きと腰の距離を変えて、すねが段に当たるか・姿勢を保てるかを一覧にする

姿勢の指定（POSES）：段の番号は 0 が一番下。None の足は「宙に上げる」で、lift_rung の段の少し上に置く。
"""

import argparse
import math
import os
import sys

import torch

import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_ladder_env import G1LadderEnv  # noqa: E402
from g1_ladder_train import get_cfgs  # noqa: E402

# 姿勢の候補
#   feet  : (左足の段, 右足の段)。None は宙に上げる足
#   lift  : 宙に上げる足を置く段（その段の上面より lift_above [m] 上、横棒の手前）
#   hands : (左手の段, 右手の段)
#   base_n: 腰をはしごの面からどれだけ離すか [m]（0.25 だと、すねが上の段に当たって足が段から外れる。0.35 が適当）
POSES = {
    "both_on_0": dict(feet=(0, 0), hands=(4, 4), base_n=0.35, desc="両足で 1 段目に立ち、両手で 5 段目を握る（学習で止まっている姿勢）"),
    "step_up": dict(feet=(0, None), lift=1, hands=(4, 4), base_n=0.35, desc="左足で 1 段目に立ち、右足を 2 段目の少し上に上げる"),
    "step_up_high_hands": dict(feet=(0, None), lift=1, hands=(5, 5), base_n=0.35, desc="同上、両手は 6 段目"),
    "on_0_and_1": dict(feet=(0, 1), hands=(5, 5), base_n=0.35, desc="左足 1 段目・右足 2 段目、両手で 6 段目"),
    "both_on_1": dict(feet=(1, 1), hands=(5, 5), base_n=0.35, desc="両足で 2 段目に立ち、両手で 6 段目を握る（1 段登った後）"),
}

FOOT_ABOVE_RUNG = 0.055  # 足が横棒に乗っているときの足首（足のリンクの原点）の高さ（横棒の中心から）[m]
FOOT_BEHIND_RUNG = 0.08  # 足首を横棒の中心からどれだけ手前に置くか（つま先側を横棒に掛ける）[m]
FOOT_WIDTH = 0.11  # 左右の足の間隔の半分 [m]
HAND_WIDTH = 0.17  # 左右の手の間隔の半分 [m]
HAND_STANDOFF = 0.04  # 手のひらを横棒の中心から、ロボットの側にどれだけ離すか [m]


def rung_point(env, k, angle):
    """段 k の中心の位置（はしごの傾き angle [rad]）。"""
    s = float(env.rung_s[k])
    return torch.tensor([s * math.cos(angle), 0.0, s * math.sin(angle)], dtype=gs.tc_float, device=gs.device)


def solve_pose(env, pose, angle):
    """腰の位置と、手足の目標位置を決め、IK で関節角度を求める。"""
    u = torch.tensor([math.cos(angle), 0.0, math.sin(angle)], dtype=gs.tc_float, device=gs.device)
    nrm = torch.tensor([-math.sin(angle), 0.0, math.cos(angle)], dtype=gs.tc_float, device=gs.device)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device)
    back = torch.tensor([-1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)

    # 足の目標位置
    feet_targets = []
    for side, k in zip((1.0, -1.0), pose["feet"]):
        if k is None:
            k_air = pose["lift"]
            p = rung_point(env, k_air, angle) + up * (FOOT_ABOVE_RUNG + 0.05) + back * (FOOT_BEHIND_RUNG + 0.05)
        else:
            p = rung_point(env, k, angle) + up * FOOT_ABOVE_RUNG + back * FOOT_BEHIND_RUNG
        p[1] = side * FOOT_WIDTH
        feet_targets.append(p)
    # 手のひらの目標位置
    hand_targets = []
    for side, k in zip((1.0, -1.0), pose["hands"]):
        p = rung_point(env, k, angle) + nrm * HAND_STANDOFF
        p[1] = side * HAND_WIDTH
        hand_targets.append(p)
    # 腰：支えている足（宙に上げていない足）の上方で、はしごの面から base_n 離す
    stand_feet = [f for f, k in zip(feet_targets, pose["feet"]) if k is not None]
    foot_z = min(float(f[2]) for f in stand_feet)  # 低い方の足が届く高さに腰を置く
    base = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
    base[2] = foot_z + 0.68
    if pose.get("base_h") is not None:
        # 水平に測る場合：腰を、低い方の足が乗る段の中心から水平に base_h 手前に置く（体はまっすぐ立てる）
        k_low = min(k for k in pose["feet"] if k is not None)
        base[0] = float(rung_point(env, k_low, angle)[0]) - pose["base_h"]
    else:
        # はしごの面からの距離：nrm・base = base_n となる x を求める（z は上で決めた値）
        base[0] = (math.cos(angle) * float(base[2]) - pose["base_n"]) / math.sin(angle)

    # 腰を固定した姿勢で IK（23 関節だけを動かす）
    robot = env.robot
    lean = 0.0 if pose.get("base_h") is not None else math.pi / 2 - angle  # はしごに合わせて体を少し前に傾ける（水平に測る場合は傾けない）
    base_quat = torch.tensor([math.cos(lean / 4), 0.0, math.sin(lean / 4), 0.0], dtype=gs.tc_float, device=gs.device)
    # 関節角度の初期値：腕を前に上げた姿勢（はしごを握る学習の「床に立って握った姿勢」と同じ）から探す
    init = torch.cat((base, base_quat, env.stand_dof_pos_all))
    robot.set_qpos(init[None])
    links = [robot.get_link(n) for n in env.env_cfg["feet_link_names"]] + [
        robot.get_link(n) for n in env.env_cfg["hand_link_names"]
    ]
    palm = env.palm_offset.cpu().numpy()
    foot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    qpos, err = robot.inverse_kinematics_multilink(
        links=links,
        poss=[t[None] for t in feet_targets + hand_targets],
        quats=[foot_quat[None], foot_quat[None], None, None],
        local_points=[None, None, palm, palm],
        init_qpos=init[None],
        dofs_idx_local=[int(i) for i in env.motors_dof_idx],
        max_samples=100,
        max_solver_iters=100,
        return_error=True,
    )
    return qpos[0], feet_targets, hand_targets, err


def check(env, name, pose, angle_deg, settle_time, verbose=True):
    angle = math.radians(angle_deg)
    env.set_ladder_angle(angle_deg)
    env.reset()
    qpos, feet_t, hand_t, err = solve_pose(env, pose, angle)
    ik_err = float(torch.norm(err[0, :, :3], dim=-1).max()) if err is not None else float("nan")

    robot = env.robot
    robot.set_qpos(qpos[None])
    robot.zero_all_dofs_velocity()
    target_motor = qpos[env.motors_dof_idx + 1][None]  # qpos は姿勢が 4 要素なので +1

    # 握る：今の手のひらの位置を、握った位置としてばねで引きつける（横棒から遠すぎたら握れない）
    palms = env._palms()
    _, _, _, dist, _ = env._ladder_coords(palms)
    gc = env.env_cfg["grip"]
    grip = dist[0] < gc["grab_distance"]
    anchor = palms.clone()
    base0 = robot.get_pos()[0].clone()
    lower, upper = robot.get_dofs_force_range(env.motors_dof_idx)
    upper = upper.reshape(-1)[: env.num_joints]

    n_steps = int(settle_time / env.dt)
    avg_from = n_steps - int(0.5 / env.dt)
    torque_sum = torch.zeros(env.num_joints, device=gs.device)
    grip_sum = torch.zeros(2, device=gs.device)
    feet_sum = torch.zeros(2, device=gs.device)
    shin_max = 0.0  # すね（膝から下のリンク。足は除く）が横棒などに当たった力の最大値
    shin_idx = [robot.get_link(n).idx_local for n in ("left_knee_link", "right_knee_link", "left_ankle_pitch_link", "right_ankle_pitch_link")]
    for i in range(n_steps):
        robot.control_dofs_position(target_motor, env.motors_dof_idx)
        palms = env._palms()
        vel = robot.get_links_vel(env.hands_links_idx)
        force = gc["stiffness"] * (anchor - palms) - gc["damping"] * vel
        norm = torch.norm(force, dim=-1, keepdim=True)
        force = force * torch.clamp(gc["max_force"] / norm.clamp(min=1e-6), max=1.0) * grip[None, :, None]
        robot.apply_links_external_wrench(force=force, links_idx_local=env.hands_links_idx)
        env.scene.step()
        shin_max = max(shin_max, float(torch.norm(robot.get_links_net_contact_force()[0, shin_idx], dim=-1).max()))
        if i >= avg_from:
            torque_sum += robot.get_dofs_control_force(env.motors_dof_idx)[0].abs()
            grip_sum += torch.norm(force[0], dim=-1)
            feet_sum += torch.norm(robot.get_links_net_contact_force()[0, env.feet_links_idx], dim=-1)
    m = n_steps - avg_from
    torque, grip_f, feet_f = torque_sum / m, grip_sum / m, feet_sum / m
    ratio = torque / upper
    base_drift = float(torch.norm(robot.get_pos()[0] - base0))
    joint_err = float((robot.get_dofs_position(env.motors_dof_idx)[0] - target_motor[0]).abs().max())
    weight = env.robot_mass * 9.81
    held = base_drift < 0.05 and joint_err < math.radians(10) and bool(ratio.max() < 0.95)
    result = dict(held=held, ik_err=ik_err, drift=base_drift, joint_err=joint_err, shin=shin_max,
                  feet=float(feet_f.sum()), grip=float(grip_f.sum()), max_ratio=float(ratio.max()),
                  max_joint=env.env_cfg["joint_names"][int(ratio.argmax())], base_x=float(base0[0]))
    if not verbose:
        return result

    print(f"\n=== {name}（{angle_deg:.0f}°）：{pose['desc']}")
    print(f"  IK の誤差（手足の目標位置との最大のずれ）: {ik_err * 100:.1f} cm   握れた手: 左 {bool(grip[0])}, 右 {bool(grip[1])}")
    print(f"  保持後：腰のずれ {base_drift * 100:.1f} cm、関節角度の最大のずれ {math.degrees(joint_err):.1f}°、"
          f"すねが当たった力の最大 {shin_max:.0f} N")
    print(f"  握る力：左 {float(grip_f[0]):.0f} N、右 {float(grip_f[1]):.0f} N（上限 {gc['max_force']:.0f} N）  "
          f"足にかかる力：左 {float(feet_f[0]):.0f} N、右 {float(feet_f[1]):.0f} N（体重 {weight:.0f} N）")
    order = torch.argsort(ratio, descending=True)
    print("  トルクの大きい関節（最大値に対する割合）:")
    for i in order[:8].tolist():
        flag = "  ← 最大値に張り付き" if ratio[i] > 0.95 else ("  ← 80% 超" if ratio[i] > 0.8 else "")
        print(f"    {env.env_cfg['joint_names'][i]:28s} {float(torque[i]):6.1f} / {float(upper[i]):5.0f} Nm ({float(ratio[i]) * 100:5.1f}%){flag}")
    print(f"  判定: {'保てる（どの関節も最大値の手前で、姿勢もほぼ崩れない）' if held else '保てない、または余裕がない'}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", type=float, nargs="+", default=[90.0, 75.0], help="はしごの傾き [deg]")
    parser.add_argument("--poses", type=str, nargs="+", default=list(POSES), choices=list(POSES), help="調べる姿勢")
    parser.add_argument("--settle-time", type=float, default=2.0, help="姿勢を保って待つ時間 [s]")
    parser.add_argument("--stiffness-scale", type=float, default=1.0, help="関節の PD のばねを何倍にして測るか")
    parser.add_argument("--viewer", action="store_true", help="画面で見る（1 つの姿勢・1 つの傾きのときに使う）")
    parser.add_argument("--base-n", type=float, nargs="+", default=None,
                        help="腰をはしごの面から離す距離 [m]（複数指定可。省略時は姿勢ごとの既定値 0.35）")
    parser.add_argument("--base-h", type=float, nargs="+", default=None,
                        help="腰を、足が乗る段の中心から水平に離す距離 [m]（複数指定可。体はまっすぐ立てる。--base-n の代わりに使う）")
    parser.add_argument("--summary", action="store_true", help="姿勢ごとの詳細を出さず、一覧表だけを表示する")
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="ロボットのモデル（assets/unitree_g1 のファイル名。例：g1_23dof_rev_1_0_sole.urdf。省略時は学習の設定どおり）",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    env_cfg, obs_cfg, reward_cfg, curriculum_cfg = get_cfgs()
    obs_cfg["add_noise"] = False
    curriculum_cfg["assist_init"] = 0.0
    env_cfg["reset"]["hang"]["prob"] = 0.0
    env_cfg["reset"]["stand_grip"]["prob"] = 0.0
    if args.urdf:
        env_cfg["urdf_path"] = args.urdf
    print(f"ロボットのモデル: {env_cfg.get('urdf_path') or 'g1_23dof_rev_1_0.urdf'}")
    rail = args.angles[0] if args.viewer else None
    env = G1LadderEnv(1, env_cfg, obs_cfg, reward_cfg, curriculum_cfg, show_viewer=args.viewer, rail_angle_deg=rail)
    if args.stiffness_scale != 1.0:
        kp = env.robot.get_dofs_kp(env.motors_dof_idx).reshape(-1)[: env.num_joints] * args.stiffness_scale
        kv = env.robot.get_dofs_kv(env.motors_dof_idx).reshape(-1)[: env.num_joints] * math.sqrt(args.stiffness_scale)
        env.robot.set_dofs_kp(kp, env.motors_dof_idx)
        env.robot.set_dofs_kv(kv, env.motors_dof_idx)
    print(f"関節の PD のばね: 学習時の {args.stiffness_scale:g} 倍。待つ時間 {args.settle_time:g} 秒（最後の 0.5 秒の平均を表示）")
    rows = []
    for angle in args.angles:
        for name in args.poses:
            dists = [("h", d) for d in args.base_h] if args.base_h else [("n", d) for d in (args.base_n or [None])]
            for mode, d in dists:
                pose = dict(POSES[name])
                if mode == "h":
                    pose["base_h"] = d
                elif d is not None:
                    pose["base_n"] = d
                r = check(env, name, pose, angle, args.settle_time, verbose=not args.summary)
                rows.append((angle, name, d if d is not None else pose["base_n"], r))
    if args.summary:
        print(f"\n腰の距離：{'足が乗る段の中心から水平に（体はまっすぐ）' if args.base_h else 'はしごの面から垂直に'}")
        print("傾き | 姿勢 | 腰の距離 | すねが当たった力の最大 | 足にかかる力 | 握る力 | 腰のずれ | 最大の関節（割合） | 判定")
        for angle, name, base_n, r in rows:
            print(f"{angle:4.0f}° | {name:18s} | {base_n:.2f} m | {r['shin']:5.0f} N | {r['feet']:5.0f} N | {r['grip']:5.0f} N | "
                  f"{r['drift'] * 100:5.1f} cm | {r['max_joint']} ({r['max_ratio'] * 100:.0f}%) | {'保てる' if r['held'] else '保てない'}"
                  + (f"  ※IK 誤差 {r['ik_err'] * 100:.0f} cm" if r["ik_err"] > 0.02 else ""))
    if args.viewer:
        print("\nビューアーを閉じると終わります。")
        try:
            while True:
                env.scene.viewer.update()
        except gs.GenesisException:
            pass


if __name__ == "__main__":
    main()
