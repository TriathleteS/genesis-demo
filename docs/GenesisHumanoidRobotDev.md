# Genesisを用いたヒューマノイドロボット開発の流れ

物理シミュレーター **Genesis** は100% Pythonで記述されているため、従来のシミュレーター（MuJoCoやIsaac Gymなど）に比べて直感的にコードを記述できます。開発は主に **「環境構築」「ロボットモデルの読み込み」「環境（タスク）の定義」「強化学習による制御モデル（Policy）の訓練」** というステップで進めます。

---

## 1. 開発環境の構築
公式には **Linux（Ubuntu）＋ NVIDIA製GPU（CUDA環境）** が最も推奨されています（並列計算やレンダリングを高速に行うため）。Mac（Apple Silicon）でも動作可能です。

Python環境（3.9以上）を用意し、本体と必要なライブラリをインストールします。

```bash
# Genesisのインストール
pip install genesis-world

# 強化学習ライブラリ（例：rsl_rl など）のインストール
pip install tensorboard
```

---

## 2. ヒューマノイドの3Dモデル（URDF/MJCF）の読み込み
Genesisでは、ロボットや障害物などのオブジェクトをすべて「**Entity（エンティティ）**」として扱います。
ロボットの記述ファイル（`.urdf` や `.mjcf`）をインポートして、初期位置や関節のパラメータを設定します。

```python
import genesis as gs

# Genesisの初期化（GPUバックエンドの指定など）
gs.init(backend=gs.gpu)

# シーン（仮想世界）の作成
scene = gs.Scene(sim_options=gs.SimOptions(dt=0.01))

# 地面の追加
plane = scene.add_entity(gs.morphs.Plane())

# ヒューマノイドロボット（URDF）の読み込みと追加
humanoid = scene.add_entity(
    gs.morphs.URDF(file="path/to/your_humanoid.urdf", pos=(0, 0, 1.0))
)

# シミュレーションのビルド
scene.build()
```

---

## 3. 関節制御（PD制御）の設定
ヒューマノイドの各関節（首、腕、腰、足など）を動かすための制御ゲイン（Pゲイン、Dゲイン）や、目標とする関節角度・トルクを設定します。

```python
# 例：すべての関節の制御モードを位置制御（Position Control）に設定
humanoid.set_control_mode(gs.control_mode.POSITION)
humanoid.set_gains(kp=40.0, kd=1.0) # PゲインとDゲインの調整
```

---

## 4. 大規模並列シミュレーション環境（Env）の作成
Genesisの最大の強みは、**数千〜数万体のロボットを同時に並列シミュレーションできる点**です。強化学習を効率的に行うために、何千もの「環境（Env）」を1つのGPU上で並列して立ち上げます。

*   **状態（Observation）の取得**: ロボットの傾き（IMU）、関節の角度、足裏の触覚・接触力（Contact Force）などを取得します。
*   **報酬（Reward）の設計**: 「前に進む」「転ばない」「エネルギーを浪費しない」といった目標を数式化してロボットに与え、望ましい行動を学習させます。

---

## 5. 強化学習（RL）による歩行・動作の訓練
並列環境を強化学習アルゴリズム（PPOなど）に繋ぎ、ひたすら歩行やタスクの訓練を回します。数万〜数十万FPSの超高速処理が可能なため、**数分〜数時間で実用レベルの歩行モーションが生成**されます。

💡 **開発をショートカットするコツ**
オープンソースのリサーチプラットフォームである **Genesis-Humanoid** などを活用すると、人間のモーションキャプチャデータからヒューマノイドへ動きをマッピングする（Retargeting）環境が最初から揃っているため、一から作る手間を省けます。

まずはリポジトリ内の `examples/locomotion` にある、4脚ロボット（Unitree Go2など）の歩行学習サンプルコード（`go2_train.py`）を動かしてみるのがおすすめです。並列学習の構造が理解できたら、ロボットのURDFモデルをヒューマノイドに差し替え、報酬関数を二足歩行用に調整していくのが最もスムーズな開発手順です。
