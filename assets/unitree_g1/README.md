# Unitree G1 のロボットモデル

`locomotion/` の学習スクリプトが読み込むヒューマノイド Unitree G1 のモデルです。

- `g1_12dof.urdf`：脚の 12 関節だけを動かせるモデル（腰・腕・頭は固定）。歩行（`g1_env.py`）で使う
- `g1_23dof_rev_1_0.urdf`：脚 12・腰 1・腕 10 の 23 関節を動かせるモデル。よじ登り（`g1_climb_env.py`）で使う
- `meshes/`：上記 URDF が参照する形状ファイル（必要なものだけを抜粋。両方の URDF で共通）
- `LICENSE`：配布元のライセンス（BSD 3-Clause License）

## 入手元

Unitree Robotics 公式の強化学習リポジトリ unitree_rl_gym から、そのまま複製しています（改変なし）。

- リポジトリ：https://github.com/unitreerobotics/unitree_rl_gym
- コミット：276801e46c5d433564f24658bac64f254b7d2d4b（2025-07-25）
- パス：`resources/robots/g1_description/`

## 読み込み時の警告について

Genesis で読み込むと、`Mesh is not watertight` や `Link 'left_ankle_pitch_link' has dubious inertia` などの警告が出ます。
URDF に書かれた質量・慣性の値が形状から推定した値と大きく違うことを知らせるものですが、
Unitree が Isaac Gym で使っているのと同じ値なので、そのまま使っています。
