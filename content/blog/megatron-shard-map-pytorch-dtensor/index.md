---
title: "Megatron-style Tensor Parallelism の型安全な実装 — JAX shard_map から PyTorch DTensor へ"
date: 2026-02-08
tags: ["LLM", "Training", "Tensor Parallelism", "PyTorch", "JAX", "DTensor", "Megatron"]
description: "Megatron-style Tensor Parallelism における通信挿入の難しさを、JAX shard_map の VMA 型システムがどう解決するかを解説し、PyTorch DTensor・DTensor Erasure・LTensor への展開を俯瞰する。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

大規模言語モデル（LLM）の学習では、モデルの重みを複数 GPU に分割する **Tensor Parallelism (TP)** が不可欠である。その代表的な実装である [Megatron-LM](https://arxiv.org/abs/1909.08053) は、Linear 層を Column-wise / Row-wise に分割し、Forward / Backward で適切な集合通信（All-Reduce, Reduce-Scatter）を挿入することで正しい勾配を得る。

しかし、この通信挿入は**手動で行う必要があり、バグの温床**となる。Forward では通信不要なのに Backward では All-Reduce が必要になるケースがあり、これを見落とすと **Silent Correctness Bug**（結果が間違っているのにエラーが出ない）が発生する。

本記事では、ezyang の [Megatron via shard_map](https://blog.ezyang.com/2026/01/megatron-via-shard-map/) を起点に、以下を解説する:

1. Megatron-style TP で Backward に隠れた All-Reduce が必要になる理由
2. JAX `shard_map` の **VMA 型システム**による型安全な解決策
3. PyTorch における対応機能: **DTensor**, **DTensor Erasure**, **LTensor**

## Megatron-style Tensor Parallelism の基本

### Column-wise Parallel Linear

Megatron の Column-wise Parallel Linear は、重み行列 $W$ を列方向に分割する。TP 度 $t$ のとき、各 GPU $i$ は $W_i \in \mathbb{R}^{h \times (h'/t)}$ を保持する。

```
Input X:  [s, b, h]      (Replicated — 全 GPU に同じデータ)
Weight W: [h, h']         → 列分割 → W_i: [h, h'/t]  (Sharded)
Output Y: [s, b, h'/t]   (Sharded — GPU ごとに異なるチャンク)
```

Forward は **通信なし**で計算できる:

$$Y_i = X \cdot W_i$$

入力 $X$ は全 GPU で同一（Replicated）なので、各 GPU が独立にローカルな行列積を実行するだけでよい。

### Backward の落とし穴: 隠れた All-Reduce

問題は Backward にある。勾配 $\frac{\partial L}{\partial X}$ を計算すると:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y_i} \cdot W_i^\top$$

各 GPU $i$ は $W_i$（重みのシャード）と $\frac{\partial L}{\partial Y_i}$（出力勾配のシャード）しか持たないため、上式で得られるのは **部分的な勾配（Partial Gradient）** にすぎない。正しい $\frac{\partial L}{\partial X}$ を得るには、全 GPU の部分勾配を足し合わせる **All-Reduce** が必要:

$$\frac{\partial L}{\partial X} = \sum_{i=0}^{t-1} \frac{\partial L}{\partial Y_i} \cdot W_i^\top$$

この All-Reduce を挿入し忘れると、各 GPU は部分勾配のみを使って更新を行い、**学習が収束しない or 精度が劣化する**。しかもエラーは出ない — これが Silent Correctness Bug である。

### Megatron の解決策: カスタム Autograd 関数

Megatron-LM では、この問題を `ColumnParallelLinear` / `RowParallelLinear` というカスタム `torch.autograd.Function` で解決している。Forward では何もせず（Identity）、Backward で All-Reduce を実行する関数を挿入する:

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_  # Forward: no-op (Identity)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)  # Backward: All-Reduce
```

このアプローチは動作するが、開発者がどの通信をどこに挿入すべきかを**手動で判断**しなければならず、新しい並列化パターンを試す際のバグリスクが高い。

## JAX shard_map と VMA 型システム

JAX の `shard_map` は、上記の問題を**型システムで防止**する。

### shard_map の基本

`shard_map` は Local SPMD（各デバイス視点でコードを記述し、明示的に集合通信を挿入するスタイル）を JAX で実現する API である。Global SPMD（`jax.Array` + 自動シャーディング伝搬）と異なり、開発者が通信を完全に制御する。

```python
@jax.shard_map(
    in_specs=(jax.P(None, None, None), jax.P(None, "tp")),
    out_specs=jax.P(None, None, "tp")
)
def colwise_linear(input, weight):
    ...
```

- `in_specs` / `out_specs` はテンソルの各次元がどのメッシュ軸にシャードされるかを指定
- `P(None, None, None)` = Replicated（全 GPU で同一）
- `P(None, "tp")` = `"tp"` 軸に沿ってシャード

### VMA（Varying Manual Axes）型システム

`shard_map` に `check_vma=True`（デフォルト）を設定すると、JAX は各テンソルが**メッシュ軸上で変動する（Varying）か不変（Invariant）か**を追跡する型システムを有効にする。

| 状態 | 意味 | 表記例 |
|---|---|---|
| Invariant | 全 GPU で値が同一 | `f32[4, 2, 8]` |
| Varying | GPU ごとに異なる値 | `f32[8, 8]{V:tp}` |

この型情報は `jax.typeof(x).vma` でアクセスできる。

### pcast による明示的な型変換

Replicated な入力と Sharded な重みの間で行列積を行う場合、VMA が異なるテンソル間の演算は不正と判定される。ここで `jax.lax.pcast` を使って明示的に型変換する:

```python
@jax.shard_map(
    in_specs=(jax.P(None, None, None), jax.P(None, "tp")),
    out_specs=jax.P(None, None, "tp")
)
def colwise_linear(input, weight):
    print('input', jax.typeof(input))
    # => input float32[4,2,8]              ← Invariant

    print('weight', jax.typeof(weight))
    # => weight float32[8,8]{V:tp}         ← Varying over tp

    input = jax.lax.pcast(input, "tp", to="varying")
    print('pcast_input', jax.typeof(input))
    # => pcast_input float32[4,2,8]{V:tp}  ← Varying にキャスト

    output = jnp.einsum("sbi,io->sbo", input, weight)
    print('output', jax.typeof(output))
    # => output float32[4,2,8]{V:tp}       ← Varying

    return output
```

`pcast` のキーポイント:

- **Forward**: No-op（何もしない）。テンソルの値は変わらない
- **Backward**: `jax.lax.psum`（All-Reduce）に変換される

これは Megatron の `_CopyToModelParallelRegion` と**同じセマンティクス**だが、JAX では `pcast` を入れ忘れると **型エラーでプログラムが停止**するため、Silent Correctness Bug を防止できる。

### なぜ型システムで防げるのか

VMA 型システムの規則は直感的である:

1. **ローカル演算の伝搬**: Varying $\times$ Varying → Varying、Invariant $\times$ Invariant → Invariant
2. **混在は不正**: Varying $\times$ Invariant → **型エラー**
3. **型変換には集合通信が必要**: `pcast(to="varying")` は Forward で Identity / Backward で All-Reduce

`input`（Invariant）と `weight`（Varying）の直接的な乗算は規則 2 により不正となるため、開発者は `pcast` の挿入を**強制**される。そして `pcast` の挿入は自動的に Backward での All-Reduce を保証する — これが型安全性の本質である。

## Global SPMD vs Local SPMD

`shard_map` の議論を深める前に、2 つの SPMD スタイルの違いを整理する。この分類は [Global vs Local SPMD](https://blog.ezyang.com/2026/01/global-vs-local-spmd/) で詳しく論じられている。

### Global SPMD（DTensor / jax.Array）

開発者はグローバルなテンソル（全体のビュー）に対してコードを記述し、フレームワークがシャーディングを自動伝搬する。

```python
# PyTorch DTensor: グローバルビューで記述
from torch.distributed.tensor import DTensor, Replicate, Shard

# 開発者はシャーディングを宣言するだけ
dtensor = DTensor.from_local(local_tensor, placement=[Shard(0)])
result = dtensor @ weight  # フレームワークが通信を自動挿入
```

**メリット**: 通信の挿入忘れが起きにくく、実験しやすい

**デメリット**: Eager モードではシャーディング伝搬のオーバーヘッドが発生（DTensor の演算は実際の計算の 7 倍以上遅いケースがある）

### Local SPMD（Megatron / shard_map）

開発者は各デバイスのローカルなテンソルに対してコードを記述し、集合通信を明示的に挿入する。

```python
# Megatron-style: ローカルビューで記述
local_output = torch.matmul(local_input, local_weight_shard)
# 開発者が手動で通信を挿入
grad_input = torch.distributed.all_reduce(grad_input)
```

**メリット**: 通信パターンを完全に制御でき、最適化の余地が大きい

**デメリット**: 通信の挿入忘れによる Silent Correctness Bug のリスク

JAX の `shard_map` は、Local SPMD のメリットを維持しつつ VMA 型システムで安全性を確保するハイブリッドアプローチと位置付けられる。

## PyTorch における Tensor Parallelism の実装

### DTensor と parallelize_module

PyTorch 2.x 系では、`torch.distributed.tensor`（DTensor）を基盤として TP を実装する API が提供されている。

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

tp_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))

# Transformer ブロックの TP plan を定義
layer_tp_plan = {
    # Attention
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
    # FFN (SwiGLU)
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}

for layer in model.layers:
    parallelize_module(layer, tp_mesh, layer_tp_plan)
```

`ColwiseParallel` / `RowwiseParallel` は内部で DTensor の `Shard` / `Replicate` placement を使い、Forward / Backward で必要な集合通信を自動挿入する。開発者は「どの層をどのスタイルで分割するか」を宣言するだけでよい。

### Sequence Parallelism との統合

Megatron-style の Sequence Parallelism も `SequenceParallel` として統合されている:

```python
from torch.distributed.tensor.parallel import SequenceParallel, PrepareModuleInput

layer_tp_plan = {
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
}
```

### 2D Parallelism: TP + FSDP

大規模クラスタでは TP（ノード内）と FSDP（ノード間）を組み合わせた 2D Parallelism を構成する:

```python
mesh_2d = init_device_mesh("cuda", (dp_size, tp_size),
                           mesh_dim_names=("dp", "tp"))

# まず TP を適用
parallelize_module(model, mesh_2d["tp"], tp_plan)

# 次に FSDP を適用
fully_shard(model, mesh=mesh_2d["dp"])
```

## DTensor Erasure: Eager オーバーヘッドの解消

### 問題: DTensor の Eager モード性能

DTensor は Global SPMD の安全性を提供するが、Eager モードでは placement の追跡・検証にオーバーヘッドが発生する。ezyang の [DTensor erasure](https://blog.ezyang.com/2026/02/dtensor-erasure/) によると、DTensor 演算は**実際の計算の 7 倍以上**遅くなるケースがある。

### 解決策: 型チェッカーとしての DTensor

DTensor Erasure のアイデアは、DTensor を**ランタイムの実行メカニズム**ではなく**動的型チェッカー**として使うことである:

1. **開発・デバッグ時**: DTensor でコードを実行し、シャーディングの正しさを検証
2. **本番学習時**: DTensor を「消去（Erase）」し、plain Tensor + 明示的な集合通信で実行

これにより、**Global SPMD の安全性**と **Local SPMD の性能**を両立する。

DTensor Erasure が成立するための条件として、以下の型システム上の要件がある:

- **Backward の勾配 placement が Forward の placement から決定論的に計算可能**であること
- **Reduced / Unreduced の状態**を区別する十分な語彙を持つこと（隠れた Backward 通信を防止）

これは JAX の VMA 型システムと**同じ問題意識**である。

## Replicate Forwards, Partial Backwards

[Replicate Forwards, Partial Backwards](https://blog.ezyang.com/2026/02/replicate-forwards-partial-backwards/) では、型システム設計のさらなる洗練が提案されている。

要点は以下の通り:

- Forward で Replicated なテンソルの勾配は、Backward では **Partial（Unreduced）** になるべき
- `pcast` / DTensor の Redistribute を明示的に呼ぶまで All-Reduce を遅延できる
- これにより、**通信の発生タイミングを開発者が完全に制御**できる

例えば TP + Sequence Parallelism の組み合わせでは、All-Reduce してから Scatter するよりも、**直接 Reduce-Scatter** する方が効率的な場合がある。Partial Backwards のセマンティクスなら、このような最適化を自然に表現できる。

## LTensor: PyTorch における shard_map 相当の実現

ezyang は [Megatron via shard_map](https://blog.ezyang.com/2026/01/megatron-via-shard-map/) の結論で、JAX の VMA 型システムに相当する機能を PyTorch DTensor に導入する計画に言及している。具体的には **LTensor** サブクラスとして、以下のメタデータを追跡する仕組みが構想されている:

| 概念 | JAX (shard_map) | PyTorch (構想) |
|---|---|---|
| ローカルテンソル | `shard_map` 内のテンソル | `LTensor` |
| 型情報 | VMA (`{V:tp}`) | placement metadata |
| Invariant → Varying | `pcast(to="varying")` | (未定: redistribute 相当) |
| 型チェック | `check_vma=True` | DTensor Erasure による検証 |

これが実現すると、PyTorch でも以下のワークフローが可能になる:

1. **DTensor（Global SPMD）で prototype を記述**  → 安全性を検証
2. **DTensor Erasure で LTensor（Local SPMD）に変換** → 性能を確保
3. **VMA 相当の型チェックで正しさを保証** → Silent Bug を防止

## まとめ

本記事の要点を整理する:

| 課題 | 解決策 | フレームワーク |
|---|---|---|
| Backward の通信挿入忘れ | VMA 型システムで型エラーにする | JAX `shard_map` |
| DTensor の Eager オーバーヘッド | DTensor Erasure（型チェッカー化） | PyTorch (提案) |
| 通信タイミングの最適化 | Partial Backwards（遅延 Reduce） | JAX / PyTorch (提案) |
| Global SPMD と Local SPMD の統合 | LTensor + DTensor 検証 | PyTorch (構想) |

Megatron-style TP の実装は、単なる行列分割にとどまらず、**Backward の通信パターンの正しさをどう保証するか**という型システムの問題に帰着する。JAX の `shard_map` + VMA は現時点でこの問題への最も洗練された解を提供しており、PyTorch も DTensor Erasure / LTensor を通じて同等の能力を獲得しつつある。

分散学習フレームワークの進化は、「より速く」だけでなく「**より安全に**」という方向にも進んでいる。

## 参考文献

- [Megatron via shard_map](https://blog.ezyang.com/2026/01/megatron-via-shard-map/) — ezyang's blog
- [Global vs Local SPMD](https://blog.ezyang.com/2026/01/global-vs-local-spmd/) — ezyang's blog
- [Replicate Forwards, Partial Backwards](https://blog.ezyang.com/2026/02/replicate-forwards-partial-backwards/) — ezyang's blog
- [DTensor erasure](https://blog.ezyang.com/2026/02/dtensor-erasure/) — ezyang's blog
- [The JAX sharding type system](https://blog.ezyang.com/2026/01/jax-sharding-type-system/) — ezyang's blog
- [Manual parallelism with shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html) — JAX Documentation
- [PyTorch Tensor Parallelism Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — PyTorch Documentation
- [torch.distributed.tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — PyTorch Documentation
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019
