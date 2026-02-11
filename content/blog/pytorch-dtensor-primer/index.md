---
title: "PyTorch DTensor 入門 — 分散テンソルの基本概念から API まで"
date: 2026-02-08
tags: ["PyTorch", "DTensor", "Distributed Training", "LLM", "SPMD", "Tensor Parallelism"]
description: "PyTorch DTensor の設計思想・コア概念（DeviceMesh, Shard, Replicate, Partial）・主要 API を初心者向けに解説し、分散学習の基盤技術を理解する。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

大規模言語モデル（LLM）の学習では、1 つの GPU にモデル全体が収まらないため、複数 GPU にモデルやデータを分散させる必要がある。Data Parallelism、Tensor Parallelism、Pipeline Parallelism など様々な分散手法が存在するが、従来の PyTorch ではこれらの手法ごとに個別の実装（`DistributedDataParallel`、`ShardedTensor` など）が必要であり、異なる並列化手法を**組み合わせる**ことが困難であった。

**DTensor**（`torch.distributed.tensor`）は、この問題を解決するために設計された PyTorch のネイティブな分散テンソルプリミティブである。DTensor は「テンソルがどのように複数デバイスに分散されているか」を抽象化し、開発者が**単一デバイスと同じセマンティクスでコードを記述**しながら、裏側では自動的に集合通信を挿入してくれる仕組みを提供する。

本記事では、PyTorch 公式の [DTensor README](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md) と PyTorch Dev Discuss での [DTensor Status, Design and Looking Forward](https://dev-discuss.pytorch.org/t/dtensor-status-design-and-looking-forward/2749) を参照しながら、DTensor の基本概念と API を初心者向けに解説する。

## DTensor が解決する課題

### 従来の分散学習の問題点

PyTorch で分散学習を行う場合、従来は以下のような課題があった:

1. **並列化手法ごとに異なる API**: DDP、FSDP、ShardedTensor など、それぞれ別の仕組みで実装されており、学習コストが高い
2. **手法の組み合わせが困難**: 例えば Tensor Parallelism（TP）と Data Parallelism（DP）を同時に使う「2D Parallelism」を構成するには、低レベルの通信コードを手書きする必要があった
3. **チェックポイントの非統一性**: 分散方法ごとに state_dict の保存・読み込み方法が異なり、学習途中での並列度変更（例: 8 GPU → 16 GPU）が困難

### DTensor のアプローチ

DTensor は、これらの課題に対して**統一された抽象化**で応える:

- 単一の `DTensor` クラスで、Sharding も Replication も表現可能
- 複数の並列化戦略を自然に**合成（compose）** できる
- state_dict が分散方法に依存しない統一的な形式になる

DTensor の公式 README では、DTensor の主要な価値として以下の 3 つが挙げられている:

> 1. **Unified Checkpointing（統一チェックポイント）**: 複雑な分散戦略をまたいで一貫した state_dict の保存・読み込みを提供
> 2. **Tensor Parallelism（テンソル並列）**: ShardedTensor よりも柔軟に、Sharding と Replication を混在させた Eager モードの TP を実現
> 3. **SPMD Foundation（SPMD 基盤）**: コンパイラベースの分散学習実装の基盤として機能

## DTensor の設計原則

DTensor Status, Design and Looking Forward の投稿では、DTensor の 4 つの設計原則が述べられている:

### 1. Simple SPMD Sharding

**SPMD（Single Program Multiple Data）** とは、すべてのデバイス（GPU）が同じプログラムを実行するが、異なるデータを処理するプログラミングモデルである。DTensor はこの SPMD モデルに基づき、「テンソルの分散方法を宣言する」だけで分散学習を記述できるようにする。

```
# SPMD の考え方
全 GPU が同じコード:  output = model(input)
データが異なる:      GPU 0 は input[:4], GPU 1 は input[4:8], ...
```

### 2. PyTorch Native Integration

DTensor は PyTorch のエコシステムに深く統合されている:

- **autograd**: 勾配計算が自動で正しく動作
- **torch.compile**: JIT コンパイルによる最適化が可能
- **tensor subclass**: `torch.Tensor` のサブクラスとして実装されており、既存の PyTorch コードと互換性がある

### 3. Single Device Semantics（単一デバイスセマンティクス）

これが DTensor の最も重要な設計原則である。開発者は**分散アルゴリズムを単一デバイスで動くかのように記述**でき、DTensor が裏側で分散に必要な通信を自動挿入する。

```python
# 通常の PyTorch コード（単一 GPU）
output = torch.matmul(input, weight)

# DTensor でも同じコード — 通信は自動
# input: Replicate, weight: Shard(1) の場合
# DTensor が内部で allgather や allreduce を挿入
output = torch.matmul(dtensor_input, dtensor_weight)
```

数学的にも、分散実行と単一デバイス実行で**同じ収束特性**が得られることが保証される。

### 4. Minimal API Philosophy

DTensor は高レベルのアルゴリズム API（例: 「FSDP を適用する」）ではなく、**最小限のプリミティブ**を提供する。これにより、システム開発者が自由に新しい並列化戦略を構築できる柔軟性を確保している。

## コア概念

DTensor を理解するための 3 つの核心概念がある: **DeviceMesh**、**Placement**、**DTensor** そのものである。

### DeviceMesh — デバイスのトポロジー

`DeviceMesh` は、利用可能なデバイス（GPU）の論理的な配置を記述するオブジェクトである。分散学習では、複数 GPU を**多次元のメッシュ（格子）** として整理することが多い。

#### 1 次元メッシュ（単純な並列化）

最も単純なケースでは、GPU を 1 列に並べる:

```python
from torch.distributed.device_mesh import init_device_mesh

# 4 GPU を 1 次元に並べる
mesh = init_device_mesh("cuda", (4,))
# mesh は [GPU0, GPU1, GPU2, GPU3] を表す
```

#### 2 次元メッシュ（多次元並列化）

TP + DP のような多次元並列化では、GPU を 2 次元格子として扱う:

```python
# 8 GPU を 2×4 の格子に配置
# 行方向: Data Parallelism (2-way)
# 列方向: Tensor Parallelism (4-way)
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

#        tp=0  tp=1  tp=2  tp=3
# dp=0 [ GPU0, GPU1, GPU2, GPU3 ]
# dp=1 [ GPU4, GPU5, GPU6, GPU7 ]

# 特定の次元のメッシュを取り出せる
tp_mesh = mesh_2d["tp"]   # TP 用のメッシュ
dp_mesh = mesh_2d["dp"]   # DP 用のメッシュ
```

DeviceMesh は内部で PyTorch の `ProcessGroup`（通信グループ）を自動的に初期化するため、開発者が `torch.distributed.new_group()` を手動で管理する必要がない。

#### なぜメッシュが必要か

物理的な GPU トポロジーを考慮してメッシュを設計することで、通信コストを最適化できる。例えば:

- **ノード内（NVLink 接続）**: 高帯域幅 → TP に適する（通信量が多い）
- **ノード間（InfiniBand 接続）**: 比較的低帯域幅 → DP/FSDP に適する（通信頻度が低い）

### Placement — テンソルの分散方法

DTensor の核心は、テンソルがメッシュ上でどのように分散されているかを記述する **Placement（配置）** である。3 種類の Placement が用意されている。

#### Shard(dim) — シャーディング

テンソルを指定した次元 `dim` に沿って分割し、各デバイスが一部分（シャード）を保持する。分割は `torch.chunk` のセマンティクスに従う。

```python
from torch.distributed.tensor import Shard

# 元のテンソル（グローバルビュー）: shape [8, 4]
# Shard(0): 0 次元目に沿って 4 分割
#
# GPU 0: [[1, 2, 3, 4],    shape [2, 4]
#          [5, 6, 7, 8]]
#
# GPU 1: [[9, 10, 11, 12],  shape [2, 4]
#          [13, 14, 15, 16]]
#
# GPU 2: [[17, 18, 19, 20], shape [2, 4]
#          [21, 22, 23, 24]]
#
# GPU 3: [[25, 26, 27, 28], shape [2, 4]
#          [29, 30, 31, 32]]
```

`Shard(1)` の場合は 1 次元目（列方向）に分割される:

```python
# Shard(1): 1 次元目に沿って 4 分割
#
# GPU 0: [[1],     shape [8, 1]
#          [5],
#          [9],
#          ...]
#
# GPU 1: [[2],     shape [8, 1]
#          [6],
#          [10],
#          ...]
```

#### Replicate() — 複製

テンソル全体のコピーを各デバイスに保持する。全 GPU が同一のデータを持つ。

```python
from torch.distributed.tensor import Replicate

# 元のテンソル: shape [8, 4]
# Replicate(): 全 GPU に同じデータ
#
# GPU 0: [[1, 2, 3, 4], [5, 6, 7, 8], ...]  shape [8, 4]
# GPU 1: [[1, 2, 3, 4], [5, 6, 7, 8], ...]  shape [8, 4]
# GPU 2: [[1, 2, 3, 4], [5, 6, 7, 8], ...]  shape [8, 4]
# GPU 3: [[1, 2, 3, 4], [5, 6, 7, 8], ...]  shape [8, 4]
```

Data Parallelism における入力データや、TP におけるシャードされない重みは Replicated な状態で保持される。

#### Partial(reduce_op) — 部分的な中間結果

各デバイスが**部分的な計算結果**を保持しており、最終的な正しい値を得るには reduce 操作（sum、avg など）が必要な状態を表す。これは行列積などの中間結果で自然に発生する。

```python
from torch.distributed.tensor import Partial

# 例: 行列積 Y = X @ W で、W が Shard(0) の場合
# 各 GPU は Y の部分和を持つ
#
# 正しい結果: Y = Y_0 + Y_1 + Y_2 + Y_3
#
# GPU 0: Y_0 (部分和)
# GPU 1: Y_1 (部分和)
# GPU 2: Y_2 (部分和)
# GPU 3: Y_3 (部分和)
#
# → all_reduce(sum) で正しい Y が得られる
```

Partial は DTensor 内部で使われる中間状態であり、通常開発者が直接作成することは少ない。しかし、DTensor のシャーディング伝搬ロジックを理解する上で非常に重要な概念である。

> **重要な制約**: DTensor の公式 README では、「DTensor cannot have placements that mix different Partial reduce types（異なる reduce 型の Partial を混在させることはできない）」と明記されている。例えば、ある次元で `Partial(sum)` かつ別の次元で `Partial(max)` とすることは不正である。これは `max` と `sum` の数学的性質が異なるため、redistribute 時に正しい結果が得られないことに起因する。

#### Placement と DeviceMesh 次元の対応

多次元メッシュの場合、Placement はメッシュの各次元に対して指定する。Placement リストの長さはメッシュの次元数と一致する:

```python
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

# placements=[Replicate(), Shard(0)] の意味:
#   dp 次元: Replicate（2 つの DP グループで同じデータ）
#   tp 次元: Shard(0)（4 つの TP グループ内で 0 次元に分割）
```

### DTensor — 分散テンソル

`DTensor` は `torch.Tensor` のサブクラスであり、以下のメタデータを保持する:

- **ローカルテンソル**: その GPU が実際に保持するデータ
- **DeviceMesh**: テンソルが分散されているデバイス群
- **Placements**: 各メッシュ次元での分散方法

```
DTensor
├── local_tensor: torch.Tensor   # この GPU のローカルデータ
├── device_mesh: DeviceMesh      # デバイスのトポロジー
└── placements: [Placement, ...]  # 各メッシュ次元の分散方法
```

DTensor は通常の `torch.Tensor` と同様に PyTorch の演算に使用できる。演算が実行されると、DTensor は内部で以下を行う:

1. **シャーディング伝搬**: 入力の Placement から出力の Placement を計算
2. **ローカル計算**: 各 GPU でローカルテンソルに対して演算を実行
3. **必要な通信の挿入**: 出力の Placement が正しくなるよう集合通信を自動挿入

## 主要 API

### distribute_tensor — テンソルの分散化

`distribute_tensor` は既存のテンソルを指定した DeviceMesh と Placement に従って分散させる。**リーフテンソル**（パラメータやバッファなど）の初期化に使う。

```python
import torch
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (4,))

# 通常のテンソルを作成（rank 0 のデータがブロードキャストされる）
tensor = torch.randn(888, 12)

# 0 次元に沿ってシャーディング
dtensor_sharded = distribute_tensor(tensor, mesh, [Shard(0)])
# 各 GPU は shape [222, 12] のシャードを保持
# dtensor_sharded.shape は (888, 12) — グローバルビュー

# 全 GPU にレプリケート
dtensor_replicated = distribute_tensor(tensor, mesh, [Replicate()])
# 各 GPU は shape [888, 12] の完全なコピーを保持
```

**注意**: `distribute_tensor` はデフォルトで rank 0 のテンソルをブロードキャストし、それを分散する。各 rank で異なるデータを持つテンソルを分散する場合は `DTensor.from_local` を使う。

### DTensor.from_local — ローカルテンソルからの変換

`DTensor.from_local` は、各 GPU が既に保持しているローカルテンソルから DTensor を構築する。`distribute_tensor` と異なり、**テンソルの再分配（通信）は行わない**。開発者が「各 GPU のローカルデータはこの Placement に従っている」と宣言する形になる。

```python
from torch.distributed.tensor import DTensor, Shard

# 各 GPU で独自のローカルテンソルを持っている場合
local_tensor = torch.randn(222, 12, device="cuda")  # 各 GPU のローカルデータ

# 「このローカルテンソルは Shard(0) の一部である」と宣言
dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
# dtensor.shape は (888, 12) — グローバルビューが復元される
```

`from_local` は autograd に対応しているため、計算途中で DTensor とローカルテンソルを行き来する場合にも勾配が正しく伝搬される。

### DTensor.to_local — ローカルテンソルへの変換

逆に、DTensor からローカルテンソルを取り出すには `to_local()` を使う:

```python
local = dtensor.to_local()  # その GPU が保持するシャードのみ取得
```

### redistribute — 分散レイアウトの変換

`redistribute` は DTensor の Placement を変更する。これが DTensor の最も強力な機能の 1 つであり、集合通信を**レイアウト変換**として抽象化する。

```python
from torch.distributed.tensor import Shard, Replicate

# Shard(0) → Replicate: 全 GPU にデータを集める
dtensor_sharded = distribute_tensor(tensor, mesh, [Shard(0)])
dtensor_replicated = dtensor_sharded.redistribute(mesh, [Replicate()])

# Shard(0) → Shard(1): シャード次元を変更
dtensor_col = dtensor_sharded.redistribute(mesh, [Shard(1)])
```

#### redistribute と集合通信の対応

redistribute の内部では、Placement の変換に応じて適切な集合通信が実行される。この対応を理解することは、DTensor の動作を把握する上で極めて重要である:

| 変換 | 集合通信 | 説明 |
|---|---|---|
| `Shard(dim)` → `Replicate()` | **all_gather** | 各 GPU のシャードを集めて全体を復元 |
| `Replicate()` → `Shard(dim)` | **ローカル chunk** | ローカルでチャンクを取るだけ（通信なし） |
| `Partial()` → `Replicate()` | **all_reduce** | 部分和を全 GPU で集約 |
| `Partial()` → `Shard(dim)` | **reduce_scatter** | 集約と分割を同時に実行 |
| `Shard(src)` → `Shard(dst)` | **all_to_all** | シャード次元を変更 |

この対応関係を図示すると:

```
        all_gather
Shard ──────────────→ Replicate
  ↑                      │
  │ reduce_scatter        │ local chunk (通信なし)
  │                      ↓
  ← ─ ─ ─ ─ ─ ─ ─ ─  Shard
        all_to_all
  (異なる dim 間)

Partial ─── all_reduce ──→ Replicate
Partial ─── reduce_scatter → Shard
```

重要なのは、**開発者は集合通信の種類を意識する必要がない**という点である。DTensor に「この Placement からあの Placement に変えたい」と指示するだけで、DTensor が適切な通信を選択・実行する。

### distribute_module — モジュールレベルの分散化

`distribute_module` は `nn.Module` 全体のパラメータを DTensor に変換するユーティリティである。`partition_fn` で各パラメータの分散方法を指定する:

```python
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, distribute_module, Shard

def partition_fn(mod_name, mod, mesh):
    """各モジュールのパラメータを分散する関数"""
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, [Shard(0)])
            )
            mod.register_parameter(name, dist_param)

# モジュール全体を分散化
model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
sharded_model = distribute_module(model, mesh, partition_fn=partition_fn)
```

## シャーディング伝搬の仕組み

DTensor の最も知的な部分は **シャーディング伝搬（Sharding Propagation）** である。PyTorch の各演算に対して、入力の Placement から出力の Placement を**自動的に推論**する。

### 具体例: 行列積

行列積 $Y = X \times W$ を例に、シャーディング伝搬を見てみよう。

#### ケース 1: X が Replicate、W が Shard(1)

```
入力 X: [s, h] — Replicate (全 GPU に同じ)
重み W: [h, h'] — Shard(1) (列方向に分割)
```

各 GPU はローカルに $Y_i = X \times W_i$ を計算する。$W_i$ は $W$ の列方向シャードなので、$Y_i$ は $Y$ の列方向シャードになる:

```
出力 Y: [s, h'] — Shard(1) ← DTensor が自動推論
```

この場合、Forward で集合通信は不要。

#### ケース 2: X が Shard(1)、W が Shard(0)

```
入力 X: [s, h] — Shard(1)  (h 次元で分割)
重み W: [h, h'] — Shard(0) (h 次元で分割 = 行方向)
```

各 GPU は $Y_i = X_i \times W_i$ を計算する。これは行列積の部分和であり、正しい $Y$ は:

$$Y = \sum_i X_i \times W_i$$

```
出力 Y: [s, h'] — Partial(sum) ← 部分和の状態
```

正しい結果を得るには `redistribute` で `Partial(sum)` → `Replicate()` に変換する（= all_reduce を実行する）必要がある。DTensor はこの必要性を**自動的に検出**し、後続の演算で必要に応じて通信を挿入する。

### 対応演算子の数

DTensor Status, Design and Looking Forward によると、DTensor は現時点で PyTorch の 2000 以上の演算子のうち約 300 をサポートしている。今後、PyTorch 2 の decomposition 機構を活用して、複雑な演算を基本演算に分解し、基本演算のシャーディング伝搬で対応する方針が示されている。

## DTensor 上に構築された並列化手法

DTensor は低レベルのプリミティブであり、その上に様々な並列化手法が構築されている。

### FSDP2（Fully Sharded Data Parallelism v2）

PyTorch の FSDP2 は DTensor をパラメータの抽象化に利用している。各パラメータは `Shard(0)` として GPU に分散され、Forward 時に `all_gather` で復元、Backward 後に `reduce_scatter` で勾配を集約する:

```python
from torch.distributed._composable.fsdp import fully_shard

# DTensor ベースの FSDP
fully_shard(model, mesh=dp_mesh)
# パラメータは Shard(0) の DTensor として管理される
```

FSDP2 が DTensor を採用した最大のメリットは、**チェックポイントの統一**と **2D/3D Parallelism との合成**が容易になったことである。

### Tensor Parallelism

Tensor Parallelism は DTensor の最も直接的な応用である。`parallelize_module` API を使い、Transformer の各層に対してシャーディング計画を定義する:

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

tp_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))

# Transformer ブロックの TP 計画
layer_tp_plan = {
    # Self-Attention
    "attention.wq": ColwiseParallel(),  # Q を列方向に分割
    "attention.wk": ColwiseParallel(),  # K を列方向に分割
    "attention.wv": ColwiseParallel(),  # V を列方向に分割
    "attention.wo": RowwiseParallel(),  # 出力射影を行方向に分割
    # FFN (SwiGLU)
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}

for layer in model.layers:
    parallelize_module(layer, tp_mesh, layer_tp_plan)
```

`ColwiseParallel` と `RowwiseParallel` は内部で DTensor の `Shard` / `Replicate` Placement を使って通信を自動管理する:

- **ColwiseParallel**: 重みを `Shard(0)` にし、出力が `Shard(-1)` になる
- **RowwiseParallel**: 重みを `Shard(1)` にし、入力の `Shard(-1)` を受け取り、出力は `Replicate()` になる（内部で all_reduce）

### 2D Parallelism: TP + FSDP の合成

DTensor の真価は、異なる並列化手法を**自然に合成**できることにある。2D Parallelism（TP + FSDP）は、多次元メッシュを使って簡潔に構成できる:

```python
# 64 GPU = 8-way DP × 8-way TP
mesh_2d = init_device_mesh("cuda", (8, 8), mesh_dim_names=("dp", "tp"))

# Step 1: TP を適用（ノード内）
parallelize_module(model, mesh_2d["tp"], tp_plan)

# Step 2: FSDP を適用（ノード間）
fully_shard(model, mesh=mesh_2d["dp"])
```

この 2 行で、各パラメータは TP 次元で `Shard` され、同時に DP 次元でも `Shard` される — つまり 2 次元の Placement を持つ DTensor となる。DTensor がメッシュの各次元の通信を独立に管理するため、複雑な通信パターンを開発者が意識する必要がない。

### Context Parallelism

Context Parallelism も DTensor の仕組みを活用している。FlashAttention との統合により、Attention の入力を系列次元に沿って `Shard` し、Ring Attention スタイルの通信を DTensor のフレームワーク内で実現する。

## 乱数生成の扱い

分散環境での乱数生成は、意外と複雑な問題である。DTensor はこの問題を `OffsetRNGTracker` で解決している:

- **Replicated なテンソル**: 全 GPU で同じ乱数を生成（同じシードを使用）
- **Sharded なテンソル**: 各 GPU で異なる乱数を生成し、それを結合すると単一デバイスで生成した場合と同じ結果になる

これにより、Dropout のような確率的操作でも**単一デバイスと同じ振る舞い**が保証される。

## カスタム演算子の登録

DTensor がサポートしていない独自の演算子に対して、シャーディング戦略を定義できる実験的な API `register_sharding` が提供されている:

```python
from torch.distributed.tensor import register_sharding

@register_sharding(my_custom_op)
def my_custom_sharding(input_placements, output_placements, mesh):
    # 入力の Placement から出力の Placement を計算するロジック
    ...
```

また、`local_map` デコレータを使うと、DTensor の自動シャーディング伝搬を一時的にバイパスし、ローカルテンソルに対して手動で集合通信を記述できる:

```python
from torch.distributed.tensor import local_map

@local_map
def manual_collective_fn(local_input):
    # ローカルテンソルに対する操作
    result = custom_operation(local_input)
    # 必要なら手動で集合通信
    torch.distributed.all_reduce(result)
    return result
```

## パフォーマンスに関する考慮事項

### Eager モードのオーバーヘッド

DTensor は `torch.Tensor` のサブクラスとして実装されているため、演算のたびに以下の追加処理が発生する:

1. 入力の Placement 情報の取得
2. シャーディング伝搬の計算
3. 出力の Placement 情報の設定

これは C++ → Python → C++ のラウンドトリップを伴い、Eager モードでは無視できないオーバーヘッドとなる。DTensor Status, Design and Looking Forward では、この問題に対する緩和策として以下が挙げられている:

- **torch.compile**: JIT コンパイルによりサブクラスのオーバーヘッドを完全に除去し、さらに演算の融合も可能にする
- **Dispatch の最適化**: OpSchema のキャッシュやハッシュの改善
- **CUDA graphs**: Eager 実行でのグラフキャプチャによるオーバーヘッド削減

### DDP/FSDP との性能差

純粋な Data Parallelism の場合、DTensor の `Replicate()` は DDP/FSDP に比べて性能が劣る可能性がある。これは DDP がモデル全体を見てバケット化や通信の重複（overlap）を最適化するのに対し、DTensor は演算子ごとにローカルに判断するためである。コンパイラベースの最適化（グラフレベルの分析、集合通信の融合）によるこの問題の改善が今後の課題として示されている。

## 今後の発展

DTensor Status, Design and Looking Forward では、以下の開発方針が示されている:

### 演算子カバレッジの拡大

PyTorch 2 の decomposition 機構を活用し、複雑な演算子を基本演算子に分解した上でシャーディング伝搬を適用するアプローチ。これにより、個々の演算子ごとにシャーディングルールを手書きする必要がなくなる。

### 不均等シャーディングのサポート

テンソルサイズが GPU 数で割り切れない場合のパディング戦略。デフォルトでパディングを行い、シャーディング伝搬時に動的にアンパディングすることで、通信とパラメータのグルーピングを簡素化する。

### 集合通信演算子の直接サポート

`functional_collective`（all_gather, reduce_scatter 等）を DTensor が直接サポートし、ProcessGroup と DeviceMesh のコミュニケータの整合性を検証する。

## まとめ

DTensor は、PyTorch における分散学習の**統一的な基盤**として設計されたプリミティブである。本記事の要点を整理する:

| 概念 | 役割 |
|---|---|
| **DeviceMesh** | GPU の論理的なトポロジーを定義し、通信グループを管理 |
| **Placement (Shard/Replicate/Partial)** | テンソルの各メッシュ次元での分散方法を記述 |
| **DTensor** | Placement メタデータ付きのテンソル。演算時にシャーディングを自動伝搬 |
| **redistribute** | Placement の変換を集合通信として抽象化（all_gather, all_reduce 等） |
| **distribute_tensor / from_local** | 通常テンソルから DTensor への変換 |
| **distribute_module / parallelize_module** | モジュール単位での分散化 |

DTensor の最大の貢献は、**集合通信をレイアウト変換として抽象化**したことにある。開発者は `all_gather` や `reduce_scatter` の名前を知らなくても、「このテンソルを Shard(0) から Replicate に変えたい」と宣言するだけで正しい通信が実行される。この抽象化により、FSDP2、Tensor Parallelism、Context Parallelism といった様々な並列化手法が DTensor という共通基盤の上に統一的に構築されている。

## 参考文献

- [DTensor Status, Design and Looking Forward](https://dev-discuss.pytorch.org/t/dtensor-status-design-and-looking-forward/2749) — PyTorch Dev Discuss
- [PyTorch DTensor README](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md) — PyTorch GitHub
- [torch.distributed.tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — PyTorch Documentation
- [Large Scale Transformer model training with Tensor Parallel](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — PyTorch Tutorial
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019
