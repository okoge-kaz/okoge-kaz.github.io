---
title: "torch.compile for Training の現在地 — 2025年8月時点の全体像"
date: 2026-02-08
tags: ["LLM", "Training", "PyTorch", "torch.compile", "Distributed Training", "DTensor"]
description: "ezyang による torch.compile の Training 向け現状レポート (2025年8月) を解説。torch.compile の基本から、DTensor・分散並列化・最適化機能・コンパイル時間問題まで、実務で必要な知識を補足しながらまとめる。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

PyTorch の `torch.compile` は、eager モードで書かれた PyTorch プログラムをコンパイルして高速化する仕組みである。2023年3月の PyTorch 2.0 で導入されて以来、推論・学習の両方で急速に採用が進んでいる。

本記事では、PyTorch コアチームの [Edward Z. Yang (ezyang)](https://blog.ezyang.com/) が2025年8月に公開した [State of torch.compile for Training (August 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) を基に、torch.compile の Training 向け機能の現在地を解説する。読者に必要な前提知識は適宜補足する。

## 前提知識: torch.compile とは何か

### Eager モード vs コンパイルモード

PyTorch は伝統的に **eager モード**で動作する。Python コードを1行ずつ即座に実行するため、デバッグが容易で柔軟性が高い。しかし、Python インタプリタのオーバーヘッドやカーネル起動のオーバーヘッドが累積し、GPU の性能を十分に引き出せない場合がある。

`torch.compile` はこの問題を解決するためのコンパイラである。デコレータを付けるだけで、PyTorch の eager コードをコンパイルして最適化できる:

```python
@torch.compile()
def f(x, y):
    ...
```

一般的に **eager モードと比較して 1.5〜2倍の高速化**が見込める。

### コンパイルの仕組み

`torch.compile` は内部的に以下の3つのコンポーネントで構成される:

1. **TorchDynamo**: Python バイトコードを解析し、PyTorch の演算をキャプチャして計算グラフ (FX Graph) を構築する
2. **AOTAutograd**: Forward の計算グラフから Backward の計算グラフを自動生成する
3. **Inductor**: 計算グラフを最適化し、GPU 向けの Triton カーネルや CPU 向けの C++/OpenMP コードを生成する

この3段構成により、ユーザが書いた eager コードをそのまま最適化できる。

## torch.compile の主要な特性

### JIT コンパイルとキャッシュ

`torch.compile` は **Just-In-Time (JIT)** コンパイラである。関数が初めて呼び出された時にコンパイルが実行される。2回目以降はコンパイル済みコードが再利用されるが、初回のコンパイルコストは避けられない。

コンパイル結果はローカルキャッシュおよびリモートキャッシュに保存でき、同じプログラムの再実行時にコンパイルをスキップできる。

### Eager モードとの互換性

`torch.compile` は PyTorch の既存エコシステムと **compositional（合成可能）** に動作する:

- **autograd**: 自動微分と組み合わせ可能（ただし double backwards は未サポート）
- **DDP / FSDP**: 分散学習フレームワークと組み合わせ可能
- **Tensor subclass**: 一部制限あり

ただし重要な制約として、**勾配の更新はコンパイルされた領域の末尾まで遅延される**。これは eager モードの autograd との統合上の制約である。

### グラフの特殊化 (Specialization) と再コンパイル

`torch.compile` は最適なコードを生成するために、**Tensor 以外の全ての引数・グローバル変数に対して積極的に特殊化**する。つまり、引数の値が変わると再コンパイルが発生する。

```python
# 例: learning rate を引数にすると、値が変わるたびに再コンパイル
@torch.compile()
def train_step(model, data, lr):
    # lr が 0.001 → 0.0001 に変わると再コンパイル発生
    ...
```

意図しない再コンパイルを検出するには以下の設定が有用:

```python
torch._dynamo.config.error_on_recompile = True
```

### Shape の扱い: 静的優先、動的にフォールバック

デフォルトでは全てのテンソルサイズを**静的に特殊化**する。シーケンス長やバッチサイズが変わると再コンパイルが発生するが、再コンパイル時にはシステムが自動的に **dynamic shapes** への切り替えを試みる。

> つまり、最初のコンパイルでは `seq_len=2048` に特殊化されたコードが生成され、`seq_len=4096` で呼び出されると再コンパイルが発生し、以降は任意の `seq_len` に対応できるコードが生成される。

### Graph Break: 非対応コードの透過的な処理

`torch.compile` がキャプチャできないコード（例: 外部ライブラリ呼び出し、複雑な制御フロー）に遭遇すると、**graph break** が発生する。graph break はデフォルトで透過的に処理され、キャプチャ可能な領域のみがコンパイルされる。

graph break を許容せずにエラーにしたい場合は `fullgraph=True` を指定する:

```python
@torch.compile(fullgraph=True)
def f(x, y):
    ...  # graph break が発生するとエラー
```

### 数値精度の差異

`torch.compile` の出力は **eager モードと bit-wise に等価ではない**。特に float16/bfloat16 のオペレーションにおいて、冗長な精度変換（downcast → upcast）がフュージョンにより省略されるため、数値が微妙に異なる場合がある。

eager モードと同じ精度の動作が必要な場合:

```python
torch._inductor.config.emulate_precision_casts = True
```

### インライン展開とループのアンローリング

`torch.compile` はデフォルトで全ての関数呼び出しをインライン展開し、ループをアンローリングする。そのため、Transformer ブロックを N 回繰り返すモデルでは、コンパイル時間が N に比例して増大する。この問題への対処は[コンパイル時間](#コンパイル時間の課題)のセクションで後述する。

## 分散並列化の現状

### DTensor: 分散テンソルの抽象化

**DTensor** は PyTorch における「グローバルテンソル」の抽象化である。テンソルのグローバルな形状を保持しつつ、実際にはローカルシャードのみをメモリに格納する。

DTensor は **Device Mesh** と **Placement** の2つの概念で分散配置を表現する:

```python
# 2D メッシュの例: Data Parallel × Tensor Parallel
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=["dp", "tp"])

# Placement の指定
# dp 軸で Replicate（全 GPU に複製）、tp 軸で Shard(0)（第0次元で分割）
placement = [Replicate(), Shard(0)]
```

#### DTensor の Placement 種別

| Placement | 意味 | 使用例 |
|-----------|------|--------|
| `Replicate()` | 全デバイスに同じデータを複製 | 入力テンソル、小さなパラメータ |
| `Shard(dim)` | 指定次元でデバイス間に分割 | TP の重み分割 |
| `Partial()` | 各デバイスが部分的な結果を持つ（リダクション待ち） | matmul 後の中間テンソル |

#### JAX との比較

JAX のシャーディング指定がテンソル中心（`P("tp", None)` でテンソルの各次元にメッシュ軸を割り当て）であるのに対し、DTensor はデバイスメッシュ中心で Placement を指定する。例えば、JAX の `P("tp", None)` に相当する DTensor の指定は、2D メッシュ `["dp", "tp"]` 上で `[Replicate(), Shard(0)]` となる。

#### DTensor の制約

- **Eager モードのオーバーヘッド**: DTensor は Python の Tensor subclass として実装されており、パラメータ数が数千に及ぶモデルではオーバーヘッドが顕著になる。キャッシュ戦略で緩和されているが、dynamic shapes では効果が薄い
- **Greedy なシャード伝播**: Eager モードの制約から、前方向のみの greedy 伝播しかできない。Backward 方向への伝播にはコンパイラの支援が必要（開発中）
- **オペレータカバレッジ**: シャード伝播ルールが定義されたオペレータのみ対応。ルールが未定義の場合は allgather でフォールバックせずエラーとなる。Transformer 系の主要オペレータ（Llama3 など）はカバー済み
- **Jagged sharding 未サポート**: Expert Parallelism でルーティングが不均衡な場合の不規則な分割には非対応
- **Dynamic shapes との相性**: `torch.compile` と DTensor を組み合わせた場合の dynamic shapes サポートは不十分（[GitHub issue #159635](https://github.com/pytorch/pytorch/issues/159635)）

#### コンパイルによる Eager オーバーヘッドの解消

DTensor を `torch.compile` と組み合わせると、Python subclass のオーバーヘッドが除去され、DTensor の操作が下位レベルのコレクティブ通信に変換（desugar）される。つまり、**DTensor の真価はコンパイルモードで発揮される**。

### Functional Collectives

DTensor を使わずに手動で SPMD プログラミングを行う場合は、**Functional Collectives** を利用できる。これは `torch.distributed` の集合通信を非破壊的（non-mutating）な関数として提供するもので、`torch.compile` との相性が良い。コンパイル時にバッファ確保を最適化（re-inplacing）できる。

ただし、**Functional Collectives は現時点で autograd をサポートしていない**。

### 分散コンパイルの課題: NCCL タイムアウト

`torch.compile` はデフォルトで **SPMD（Single Program, Multiple Data）を仮定しない**。つまり、各ランクが独立にコンパイルを行うため、ランク間でコンパイラの判断が異なると（例: あるランクでは graph break が発生し別のランクでは発生しない）、集合通信の不一致により **NCCL タイムアウト**が発生するリスクがある。

対策:
- ランク間で divergent な動作（ランクに依存する条件分岐など）を排除する
- communicative な collective を graph break 点に挿入して同期を取る

## 最適化機能

### Inductor バックエンド

**Inductor** は `torch.compile` のデフォルトバックエンドで、以下の最適化を行う:

- **Triton カーネル生成**: pointwise 演算・reduction の自動フュージョン
- **Matmul へのフュージョン**: pointwise 演算と matmul の融合（例: bias 加算、activation function）
- **Matmul オートチューニング**: cuBLAS, CUTLASS, Triton の3つのバックエンドから最適なものを自動選択

### CUDA Graphs サポート

`torch.compile` は **CUDA Graphs** を内蔵しており、手動で CUDA Graphs を構成するよりも健全性の保証（soundness guarantee）が優れている。CUDA Graphs はカーネル起動のオーバーヘッドを排除し、特にカーネル数が多い学習ループで効果的である。

> **CUDA Graphs とは**: GPU カーネルの起動シーケンスを記録し、1回の API 呼び出しで一括再生する仕組み。カーネルの計算時間に対して起動オーバーヘッドが支配的な場合に大幅な高速化が得られる。

### 自動 Activation Checkpointing

`torch.compile` は **Activation Checkpointing（勾配チェックポインティング）** の自動最適化を提供する。

> **Activation Checkpointing とは**: Forward パスで計算された中間結果（activation）を全て保存する代わりに、一部を破棄して Backward パスで再計算することでメモリ使用量を削減する技術。メモリと計算のトレードオフである。

PyTorch の eager API（`torch.utils.checkpoint`）では、どの層をチェックポイントするかを手動で指定する必要がある。一方、`torch.compile` の自動 Activation Checkpointing は**メモリ-計算トレードオフのグローバル最適化**を行い、手動指定よりも効率的な戦略を自動的に発見できる。

ただし、ハイパーパラメータの調整が難しく、既知のバグも報告されている点に注意が必要。

### FP8 最適化

FP8（8ビット浮動小数点）による学習の最適化がサポートされており、[torchao](https://github.com/pytorch/ao) にアップストリームされている。FP8 は H100 以降の GPU で利用可能で、メモリ使用量と通信量を削減しつつ学習スループットを向上させる。

### Flex Attention

**Flex Attention** は `torch.compile` と統合された柔軟な Attention 実装である。2025年1月時点の125リポジトリから、2025年8月時点で **632リポジトリ**に採用が拡大している。

Flex Attention は以下のような多様な Attention パターンを効率的に実装できる:

- Chunked Attention
- Document Masking（複数ドキュメントのパッキング時のマスク処理）
- Context Parallelism（長いシーケンスを複数 GPU に分割）

研究用途では非常に有用なツールだが、数値結果が標準的な Attention 実装と微妙に異なる場合がある点に留意する必要がある。

### Helion: 高レベル Triton プログラミング

**Helion** は Triton カーネルをより高レベルに記述するためのプロジェクトで、PyTorch の eager モードに近い記法で GPU カーネルを書けることを目指している。2025年10月の Beta リリースを目標に開発中だが、**まだ Production Ready ではない**。

## コンパイル時間の課題

### JIT コンパイルの問題

JIT コンパイルは GPU クラスタ上で実行されるため、**コンパイル中は GPU を学習に使えない**。大規模モデルではコンパイルに数分〜数十分かかることがあり、GPU リソースの浪費となる。

### 再コンパイルの原因

- **Dynamic shapes**: バッチサイズやシーケンス長が変動する場合
- **Shape に依存する条件分岐**: テンソルのサイズによって異なるパスを通るコード

### Transformer ブロックの最適化

前述の通り、デフォルトでは N 層の Transformer を全てアンローリングしてコンパイルする。**Transformer ブロック単体をコンパイルする**ことで、コンパイル時間を大幅に削減できる:

```python
# 悪い例: モデル全体をコンパイル → N 層分のコンパイル時間
model = torch.compile(model)

# 良い例: Transformer ブロックのみをコンパイル → 1 層分のコンパイル時間
for layer in model.layers:
    layer = torch.compile(layer)
```

### Precompile: AOT コンパイルへの移行

PyTorch チームは JIT コンパイルのキャッシュを**長期的な解決策とは考えていない**。代わりに、**precompile** — Ahead-of-Time (AOT) コンパイルのワークフローを開発中である。

Precompile は以下の手順で動作する:

1. 学習スクリプトから事前にコンパイルを実行
2. コンパイル結果をバイナリとして保存
3. 学習時にバイナリを直接ロードして実行

これは **AOTInductor** の ABI 安定インターフェースを活用しており、PyTorch のバージョンが変わっても同じバイナリを利用できる可能性がある。

## 先進的な並列化への展望

### SimpleFSDP

**SimpleFSDP** は FSDP パターンの集合通信を自動挿入し、FSDP に特化した最適化パスを適用するフレームワークである。スコープを限定することで、より確実な最適化を実現する。

### AutoParallel

**AutoParallel** は [GSPMD](https://arxiv.org/abs/2105.04663) スタイルの自動シャーディング発見を目指すプロジェクトで、ユーザが単一ノード向けに書いたコードから、Data Parallelism・Tensor Parallelism・Expert Parallelism の最適な組み合わせを自動的に決定する。

### JAX との哲学的な違い

JAX は汎用的なソルバ（GSPMD）からスタートし、後から手動のエスケープハッチ（`shard_map` など）を追加している。一方、PyTorch は手動パターン（DDP, FSDP, Megatron-style TP）からスタートし、後から自動化（AutoParallel）を追加している。アプローチは逆だが、両者は収束しつつある。

## 実践的な導入ガイド

### TorchTitan からの出発

torch.compile を大規模学習に導入する際の推奨パターンは、**[torchtitan](https://github.com/pytorch/torchtitan) をフォーク**して自身の学習基盤とすることである。

torchtitan は PyTorch のネイティブ機能を統合した参照実装であり、以下の機能を実演する:

- `torch.compile` による学習ループの最適化
- FSDP2 / Tensor Parallelism の統合
- Activation Checkpointing の設定
- 分散学習のベストプラクティス

自身のモデルや学習設定に合わせて、torchtitan のコンポーネントを段階的に差し替えていくのが実務的なアプローチである。

### 導入時のチェックリスト

1. **まず torchtitan を動かす**: 参照実装でベースラインを確認
2. **Transformer ブロック単位でコンパイル**: コンパイル時間を削減
3. **`error_on_recompile` を有効化**: 意図しない再コンパイルを早期検出
4. **`fullgraph=True` を試す**: graph break の有無を確認
5. **数値精度を検証**: eager モードとの差異を許容できるか確認
6. **分散環境でのテスト**: NCCL タイムアウトが発生しないか検証

## まとめ

`torch.compile` は Training 向けのコンパイルインフラとして成熟しつつあり、1.5〜2倍の高速化を PyTorch の既存エコシステムと組み合わせて実現できる。DTensor による分散テンソルの抽象化、Inductor による自動カーネル生成、自動 Activation Checkpointing などの最適化機能が揃っている。

一方で、以下の課題が残されている:

- **コンパイル時間**: JIT コンパイルによる GPU クラスタのアイドル時間（precompile で解決予定）
- **Dynamic shapes**: DTensor との組み合わせでの制約
- **分散コンパイルの同期**: ランク間の divergent な動作による NCCL タイムアウトリスク
- **数値精度**: eager モードとの bit-wise な差異

大規模学習への導入を検討するチームは、torchtitan をフォークし、段階的に自身のワークロードに適応させるアプローチが推奨される。

## References

- [State of torch.compile for Training (August 2025)](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) — Edward Z. Yang
- [torchtitan](https://github.com/pytorch/torchtitan) — PyTorch reference training implementation
- [PyTorch 2.0: torch.compile](https://pytorch.org/get-started/pytorch-2.0/) — Official documentation
- [DTensor RFC](https://github.com/pytorch/pytorch/issues/88838) — PyTorch DTensor design discussion
- [torchao](https://github.com/pytorch/ao) — PyTorch Architecture Optimization
- [GSPMD: General and Scalable Parallelization for ML Graphs](https://arxiv.org/abs/2105.04663) — Xu et al., 2021
