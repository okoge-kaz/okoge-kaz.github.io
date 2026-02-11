---
title: "Context Parallelism と Ring Attention — 長系列学習のための系列並列化"
date: 2026-02-08
tags: ["LLM", "Training", "Parallelism", "Ring Attention", "Context Parallelism", "Long Context"]
description: "LLM の長系列学習を実現する Context Parallelism の仕組みを、Ring Attention の原理から Causal Masking の負荷分散戦略まで解説する。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

LLM の学習・推論において、コンテキスト長の拡大は重要な研究課題である。しかし、Transformer の Self-Attention は系列長 $s$ に対して $O(s^2)$ のメモリと計算量を要するため、単一 GPU のメモリに収まる系列長には厳しい上限がある。

例えば 1 億トークンの系列を処理するには、Attention の中間テンソルだけで 1000 GB 以上のメモリが必要となり、現行 GPU（A100: 80GB, H100: 80GB）では到底扱えない。

**Context Parallelism (CP)** は、入力系列をシーケンス次元に沿って複数 GPU に分割し、各 GPU が系列の一部分のみを担当することでこの制約を打破する手法である。本記事では、CP の代表的な実現手法である **Ring Attention** を中心に、その原理・通信パターン・負荷分散戦略を解説する。

## Context Parallelism の位置づけ

LLM の大規模学習では、複数の並列化戦略を組み合わせて利用する。

| 並列化手法 | 分割対象 | 主な目的 |
|---|---|---|
| Data Parallelism (DP) | データ（バッチ） | スループット向上 |
| Tensor Parallelism (TP) | モデルの重み（層内） | モデルメモリ削減 |
| Pipeline Parallelism (PP) | モデルの層（層間） | モデルメモリ削減 |
| Sequence Parallelism (SP) | Activation（LayerNorm/Dropout 部分） | Activation メモリ削減 |
| **Context Parallelism (CP)** | **入力系列（シーケンス次元）** | **長系列対応** |

Megatron-LM における Sequence Parallelism (SP) は TP と組み合わせて LayerNorm や Dropout の Activation を分割するものであり、Attention 計算自体は分割しない。一方 CP は **Attention を含む全レイヤーの Activation をシーケンス次元に沿って分割**する点が本質的に異なる。

総 GPU 数は以下の関係で決まる:

$$\text{Total GPUs} = \text{TP} \times \text{CP} \times \text{PP} \times \text{DP}$$

## Ring Attention の原理

Ring Attention ([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) は、CP を実現する代表的なアルゴリズムである。

### Blockwise Attention

Ring Attention の基盤は **Blockwise Attention**（FlashAttention と同様のブロック単位計算）である。通常の Self-Attention は以下のように計算される:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

この計算を一度に行うと $O(s^2)$ のメモリが必要だが、Query をブロック単位で処理し、各ブロックの softmax 統計量（max と sum）を逐次的に更新（online softmax）することで、**中間テンソル $QK^\top$ 全体をメモリに保持する必要がなくなる**。

重要な性質として、Self-Attention は **Key-Value ブロックの処理順序に依存しない**（permutation invariance）。すなわち、KV ブロックをどの順番で処理しても、各ブロックの統計量を正しく結合すれば最終結果は同一になる。Ring Attention はこの性質を活用する。

### Ring 通信トポロジー

$N$ 台の GPU をリング状に接続し、入力系列を $N$ 個のチャンクに分割する。各 GPU $i$ は以下を保持する:

- **Query ブロック** $Q_i$: 自身が担当する系列チャンクの Query（固定）
- **KV ブロック**: リング上を循環する Key-Value ペア

アルゴリズムは $N$ ステップで構成される:

```
for step in range(N):
    # 1. 現在保持している KV ブロックで Blockwise Attention を計算
    local_attn = blockwise_attention(Q_i, K_current, V_current)

    # 2. 統計量を更新（online softmax の逐次結合）
    output_i = combine(output_i, local_attn)

    # 3. KV ブロックをリング上の次の GPU に送信（同時に前の GPU から受信）
    K_current, V_current = ring_send_recv(K_current, V_current)
```

各ステップで GPU $i$ は:
- KV ブロックを GPU $(i+1) \mod N$ に **送信**
- KV ブロックを GPU $(i-1) \mod N$ から **受信**

$N$ ステップ後、各 GPU の Query は全系列の KV と Attention を計算し終え、正確な結果が得られる。

### 通信と計算のオーバーラップ

Ring Attention の核心的な工夫は、**KV ブロックの送受信と Attention 計算を同時に実行**することで通信コストを隠蔽する点にある。

通信コストが計算コストに隠れる条件は:

$$c \geq \frac{F}{B}$$

ここで $c$ はブロックサイズ（トークン数）、$F$ は GPU の演算性能（FLOPS）、$B$ はデバイス間帯域幅である。ブロックサイズが十分に大きければ、通信は計算の裏で完了し、**実質的な通信オーバーヘッドはゼロ**となる。

### メモリ使用量

Ring Attention のメモリ使用量は以下の通りである:

| コンポーネント | メモリ |
|---|---|
| Query ブロック | $bch$ |
| 現在の KV ブロック（K, V） | $2bch$ |
| 受信バッファ（K, V） | $2bch$ |
| 出力ブロック | $bch$ |
| **合計** | **$6bch$** |

ここで $b$ はバッチサイズ、$c$ はブロックサイズ、$h$ は hidden dimension である。**メモリ使用量は系列長 $s$ に依存しない**。これにより、GPU 数に比例して系列長をスケールできる:

> GPU $N$ 台で Blockwise Attention が系列長 $s$ を処理できるなら、Ring Attention では系列長 $N \times s$ を処理できる。

## Causal Masking と負荷分散問題

### 問題: Naive 分割の計算量不均衡

Causal（自己回帰）モデルでは、位置 $i$ のトークンは位置 $j \leq i$ のトークンにのみ Attend できる。系列を単純に前半・後半に分割すると、深刻な負荷不均衡が生じる。

4 GPU に系列 $[t_0, t_1, ..., t_{15}]$ を naive に分割した場合:

```
GPU 0: [t0,  t1,  t2,  t3 ]  → Q が attend できる KV が少ない（計算量: 小）
GPU 1: [t4,  t5,  t6,  t7 ]  → 計算量: 中
GPU 2: [t8,  t9,  t10, t11]  → 計算量: 大
GPU 3: [t12, t13, t14, t15]  → Q が全 KV に attend（計算量: 最大）
```

GPU 0 は causal mask により大部分の KV との Attention がマスクされ計算量が少ない一方、GPU 3 はほぼ全てのKV と Attention を計算する必要がある。Ring Attention のステップ数は全 GPU で同一であるため、計算量が少ない GPU はアイドル時間が生じ、全体の効率が低下する。

### 解決策 1: Striped Attention

[Striped Attention (Brandon et al., 2023)](https://arxiv.org/abs/2311.09431) は、トークンを **ラウンドロビン方式** で各 GPU に分配する:

```
GPU 0: [t0, t4, t8,  t12]
GPU 1: [t1, t5, t9,  t13]
GPU 2: [t2, t6, t10, t14]
GPU 3: [t3, t7, t11, t15]
```

各 GPU が系列全体から均等にトークンを受け取るため、causal mask による計算量の偏りが大幅に緩和される。

Attention 計算においてトークンの物理的な配置順序は結果に影響しない（位置情報は Positional Encoding で保持される）ため、この再配置は Attention の正確性を損なわない。

### 解決策 2: Zigzag 分割

Zigzag 分割は、系列の先頭と末尾のトークンを組み合わせて各 GPU に配分する:

```
GPU 0: [t0, t7,  t8,  t15]  → 先頭（計算量小）+ 末尾（計算量大）
GPU 1: [t1, t6,  t9,  t14]
GPU 2: [t2, t5,  t10, t13]
GPU 3: [t3, t4,  t11, t12]
```

計算量の多いトークンと少ないトークンをペアリングすることで、各 GPU の総計算量を均等化する。Striped と比べてメモリアクセスの局所性が若干良い場合がある。

### 性能比較

| 分割戦略 | 負荷分散 | 実装の容易さ | 備考 |
|---|---|---|---|
| Naive | 悪い | 容易 | Causal モデルでは非推奨 |
| Striped | 良い | 中程度 | DeepSpeed 等で採用 |
| Zigzag | 良い | 中程度 | Megatron-LM で採用 |

実測では Zigzag と Striped の性能差は系列長が長くなるほど小さくなる。Megatron-LM の CP 実装では、さらに causal mask でマスクされる領域の計算自体をスキップする最適化も行われており、基本的な Ring Attention よりも高い効率を達成している。

## Ulysses: もう一つの CP 実現手法

[DeepSpeed-Ulysses (Jacobs et al., 2023)](https://arxiv.org/abs/2309.14509) は Ring Attention とは異なるアプローチで CP を実現する。

### 動作原理

1. 入力系列を $N$ GPU にトークン次元で分割
2. **All-to-All 通信**で、トークン次元の分割から **Attention Head 次元の分割**に変換
3. 各 GPU は全トークンに対して一部の Head の Attention を計算（FlashAttention を適用可能）
4. 再び All-to-All 通信でトークン次元の分割に戻す

### Ring Attention との比較

| | Ring Attention | Ulysses |
|---|---|---|
| 通信パターン | Point-to-Point（リング） | All-to-All |
| 通信回数 | $N$ 回（ステップごと） | 2 回（前後） |
| 最大並列度 | 制限なし | Attention Head 数が上限 |
| GQA/MQA との相性 | 良い（KV のみ循環） | 制約あり（KV Head 数が少ない） |
| 通信量 | $O(s \cdot d / N)$ per step | $O(s \cdot d)$ per all-to-all |

Ulysses は通信回数が少なく、NVLink/NVSwitch のような高帯域接続では Ring Attention より高速な場合がある。一方、GQA/MQA のように KV Head 数が少ないアーキテクチャでは並列度が制限されるため、Ring Attention の方が適している。

実用的には、**Hybrid CP**（ノード内で Ulysses、ノード間で Ring Attention）を採用する手法 ([USP](https://github.com/feifeibear/long-context-attention)) も提案されている。

## Megatron-LM における実装

Megatron-LM では `context_parallel_size` を設定するだけで CP を有効化できる:

```bash
--context-parallel-size 8
```

内部的には、All-Gather / Reduce-Scatter をリングトポロジー上の Point-to-Point 通信に変換し、FlashAttention / cuDNN Flash Attention カーネルと組み合わせて実行する。Causal mask で不要となる下三角領域の計算をスキップし、GPU 間の最適な負荷分散を実現している。

対応する Attention タイプ:
- MHA / MQA / GQA
- Uni-directional（causal）/ Bi-directional masking

要件: Megatron-Core >= 0.5.0, Transformer Engine >= 1.1

## まとめ

- **Context Parallelism** は系列長方向にモデルの Activation を分割し、長系列の学習・推論を可能にする
- **Ring Attention** はリング通信で KV ブロックを循環させ、通信と計算のオーバーラップにより実質ゼロオーバーヘッドを達成する
- メモリ使用量は系列長に依存せず、GPU 数 $N$ 倍の系列長をスケールできる
- **Causal モデルでは負荷分散が課題**であり、Striped / Zigzag 分割で解決する
- **Ulysses** は All-to-All 通信ベースの代替手法で、高帯域環境で有利
- 実用上は **Hybrid CP**（Ulysses + Ring Attention）や、他の並列化手法（TP, PP, DP）との組み合わせが重要

---

## 参考文献

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) — Liu et al., 2023
- [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431) — Brandon et al., 2023
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509) — Jacobs et al., 2023
- [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://github.com/feifeibear/long-context-attention) — Fang et al., 2024
- [Megatron-Core Context Parallelism Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)
- [Introducing Context Parallelism — Insu Jang](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
