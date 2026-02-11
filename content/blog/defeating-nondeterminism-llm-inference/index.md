---
title: "LLM推論における非決定性の克服 — Batch Invariance とRLVRへの影響"
draft: true
date: 2026-02-08
tags: ["LLM", "Inference", "RLVR", "Reinforcement Learning", "vLLM", "SGLang", "Determinism"]
description: "LLM推論で temperature=0 でも出力が変わる根本原因（Batch Invariance の欠如）と、それが RLVR（Reinforcement Learning with Verifiable Rewards）に与える深刻な影響を解説する。"
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

> 本記事は [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Thinking Machines) の内容をベースに、RLVR における課題を踏まえて詳細に解説したものである。

## はじめに

LLM 推論において `temperature=0`（greedy decoding）を設定すれば、出力は決定的になるはずである。しかし実際には、**同一プロンプトに対して異なる出力が生成される**ことが広く知られている。これは商用 API（OpenAI, Anthropic 等）だけでなく、vLLM や SGLang などのオープンソース推論エンジンでも発生する。

この非決定性は単なる不便さにとどまらず、**RLVR（Reinforcement Learning with Verifiable Rewards）において深刻な問題**を引き起こす。本記事では、非決定性の根本原因を技術的に解説し、その解決策と RLVR への影響を詳しく述べる。

## 「よくある誤解」— 浮動小数点の非結合性

### 浮動小数点演算の性質

LLM 推論の非決定性について議論する際、最もよく挙げられる原因は「GPU の並列処理と浮動小数点演算の非結合性（non-associativity）」である。

浮動小数点演算では結合法則が成立しない:

$$(a + b) + c \neq a + (b + c)$$

具体例を示す:

```
(0.1 + 1e20) - 1e20 = 0       // 0.1 が精度落ちで消失
0.1 + (1e20 - 1e20) = 0.1     // 先に大きい数同士が打ち消し合う
```

実際に `[1e-10, 1e-5, 1e-2, 1]` とそれぞれの符号反転値を含む配列に対して、10,000 回ランダムな順序で総和を計算すると、**102 個の異なる結果**が得られる。

### なぜこの説明では不十分か

しかし、この「浮動小数点 + 並列処理」という説明は**不完全**である。同一の行列積を GPU 上で繰り返し実行した場合、**ビット単位で完全に一致する結果**が得られる。つまり、個々のカーネル実行は run-to-run deterministic（同一入力に対して同一出力）である。

では、なぜ推論システム全体としては非決定的になるのか？

## 根本原因: Batch Invariance の欠如

### 問題の本質

真の原因は **Batch Invariance（バッチ不変性）の欠如**である。

推論カーネルは個々の forward pass については決定的であるが、**バッチサイズに応じて異なる出力を生成する**。つまり、同じ入力であっても、他のリクエストと一緒にバッチ処理されるか、単独で処理されるかによって結果が変わる。

推論サーバーはリクエストの負荷に応じて動的にバッチサイズを変更する（dynamic batching / continuous batching）。ユーザーの視点では、**サーバー負荷という制御不能な変数がバッチサイズに変換され、それが出力に影響する**。

これを形式的に表現すると:

> Batch invariance を持たないカーネルの性質（バッチサイズ依存性）と、バッチサイズの非決定性（サーバー負荷）を合成すると、**非決定的なシステム**が出来上がる。

### 実験的証拠

`Qwen/Qwen3-235B-A22B-Instruct-2507` に対して "Tell me about Richard Feynman" というプロンプトを `temperature=0` で 1,000 回サンプリングした結果:

- **通常のカーネル**: **80 個の異なる completion** が生成された。最頻出の completion でも出現回数は 78 回に過ぎない。最初の分岐はトークン 103 で発生し、1,000 件中 992 件は "Queens, New York" を、8 件は "New York City" を生成した。
- **Batch invariant カーネル**: **1,000 件すべてが完全に一致**した。

## 各カーネルにおける Batch Invariance の確保

Batch invariance を確保するためには、RMSNorm、行列積、Attention の3つの主要な演算それぞれにおいて対策が必要である。

### RMSNorm

#### 計算式

$$\text{RMSNorm}(x) = x \cdot \text{rsqrt}(\text{mean}(x^2)) \cdot w$$

#### Data-Parallel 戦略

RMSNorm は各バッチ要素に対する reduction（平均計算）を含む。Batch invariance を保つには、**1つのコアに1つのバッチ要素を割り当て**、reduction の順序を固定する。

バッチサイズが増加した場合、各コアが逐次的により多くの行を処理するため、reduction の順序は変わらない。しかし、**バッチサイズが小さい場合**に問題が発生する。コア数がバッチ要素数を超えると、1つの要素の reduction を複数コアで分割（split reduction）する必要が生じ、これが accumulation order を変化させて batch invariance を破壊する。

#### 対策

小さなバッチサイズでも一貫した reduction 戦略を用いる。性能低下を受け入れる必要があるが、決定性が保証される。

### 行列積（MatMul）

#### 通常の並列化

標準的な行列積の並列化では、出力行列を 2D タイルに分割し、各コアに割り当てる。各コアが reduction 次元全体を独立に処理するため、batch invariance は自然に保たれる。

#### Split-K の問題

小さなバッチサイズでは、出力タイル数が GPU コア数を下回り、並列度が不足する。これを解消するために **Split-K** 戦略が使われる。Split-K は reduction 次元を複数のコアで分割して並列処理するが、**部分和の accumulation order がバッチサイズに依存して変化する**ため、batch invariance が破壊される。

#### Tensor Core の影響

さらに、バッチサイズの変化により異なる Tensor Core 命令（例: `wgmma.mma_async.sync.aligned.m64n128k16`）が選択されることがある。異なる命令は内部的に異なる reduction order を持つため、これも batch invariance を破壊する。

#### 対策

全ての入力形状に対して**固定のカーネル構成**を使用する。cuBLAS と比較して約 **20% の性能低下**が生じるが、batch invariance は保証される。性能低下は主に、非常に小さなバッチサイズでの並列度不足と、タイルの量子化効果（"jigsaw pattern"）に起因する。

実験として `[2048, 4096] × [4096, 4096]` の行列積で、単一要素とバッチ処理時の差の最大値は **1669.25** であった。

### Attention

Attention は最も困難な演算である。RMSNorm や行列積と異なり、**特徴次元とシーケンス次元の両方に対する reduction** が必要であり、さらに推論固有の最適化（chunked prefill、prefix caching）が invariance を破壊する可能性がある。

#### 境界条件の問題

KV キャッシュの明示的な分離が batch invariance を破壊する。例えば、ブロックサイズ 32 で 80 トークンのキャッシュに 48 トークンの新規入力がある場合:

- キャッシュ部分: 3 ブロック（80 トークン）
- 新規部分: 2 ブロック（48 トークン）
- 合計: 5 ブロック（128 トークン）

この分割パターンは新規トークン数に依存するため、同じ合計長でも処理方法が変わってしまう。

**対策**: Attention カーネル実行前に KV キャッシュとページテーブルを更新し、新旧の区別なく一貫したレイアウトで処理する。

#### Split-KV / FlashDecoding の問題

Decode フェーズでは Query 長が 1 になるため並列度が不足する。一般的な対策として KV 次元で分割する Split-KV（FlashDecoding）が使われるが、**分割数がバッチサイズに応じて変化する**ため batch invariance が破壊される。

**対策**: 分割数を固定するのではなく、**分割サイズを固定**する。例えば KV 長 1,000 の場合、分割サイズ 256 で `[256, 256, 256, 232]` の 4 分割となる。この分割パターンは同時処理されるトークン数に依存しないため、batch invariance が保たれる。

#### Chunked Prefill と Prefix Caching

Chunked Prefill（長い Prefill を複数チャンクに分割して処理）と Prefix Caching（共通プレフィックスのキャッシュ共有）は、Attention の計算境界を変化させるため、注意深く実装しなければ batch invariance を破壊する。

## 実装

### batch-invariant-ops リポジトリ

著者らは `thinking-machines-lab/batch-invariant-ops` として参照実装を公開している。

`torch.Library` を用いて PyTorch のオペレータを非侵襲的に置換する方式を採用しており、vLLM の FlexAttention バックエンドと統合して決定的推論を実現している。

固定分割サイズの Attention 戦略に必要な FlexAttention 内部の変更は「近日中にアップストリームに反映予定」とされている。

### 性能ベンチマーク

単一 GPU で `Qwen-3-8B` を用い、1,000 シーケンス（出力 90-110 トークン）を処理した結果:

| 構成 | 処理時間 |
|------|----------|
| vLLM デフォルト | 26 秒 |
| 未最適化の決定的 vLLM | 55 秒 |
| 改善された Attention カーネル | 42 秒 |

デフォルトに対して約 1.6 倍の遅延が残るが、未最適化版の 2.1 倍からは大幅に改善されている。残りの遅延は FlexAttention の vLLM 統合が未最適化であることに起因する。

## RLVR への影響

ここまでの議論は推論の正確性に関するものだったが、非決定性が最も深刻な影響を及ぼすのは **RLVR（Reinforcement Learning with Verifiable Rewards）** の文脈である。

### On-Policy vs Off-Policy の問題

RLVR（GRPO 等）では、モデルが自身でサンプリングした出力に対して報酬を計算し、ポリシーを更新する。ここで重要なのは **on-policy**（学習中のポリシーと同一のポリシーからサンプリング）と **off-policy**（異なるポリシーからサンプリング）の区別である。

#### 非決定性が引き起こす暗黙的な Off-Policy 化

典型的な RLVR の実装では、**サンプリング（推論）と学習（訓練）を異なるプロセスで行う**。推論エンジン（vLLM 等）がサンプリングを行い、学習フレームワーク（Megatron-LM, DeepSpeed 等）がポリシー更新を行う。

ここで問題が発生する。推論エンジンと学習フレームワークは**異なる数値計算パスを持つ**ため、同一の重みから出発しても、同一の入力に対して**異なる logits を出力する**。つまり：

$$\pi_{\text{sampler}}(\cdot | x) \neq \pi_{\text{trainer}}(\cdot | x)$$

これは **on-policy のつもりで行っている学習が、実際には off-policy になっている**ことを意味する。サンプラーとトレーナーの間に分布のずれ（distribution shift）が存在し、KL ダイバージェンスが非ゼロとなる。

### なぜ Off-Policy 化が問題なのか

On-policy アルゴリズム（PPO, GRPO 等）は、現在のポリシーからサンプリングされたデータで学習することを前提として設計されている。Off-policy データで学習すると:

1. **報酬の崩壊**: 学習が進むにつれて報酬が急激に低下する可能性がある
2. **KL ダイバージェンスのスパイク**: サンプラーとトレーナーの分布の乖離が突然増大し、学習が不安定化する
3. **損失関数のスパイク**: KL ダイバージェンスのスパイクに伴い、損失が急激に増加する

### 実験結果

著者らは BigMath データセットを用い、`Qwen 2.5-VL Instruct 8B` を初期ポリシーとして、最大ロールアウト長 4,096 で RLVR 実験を行った。3つの構成を比較している:

#### 1. Off-Policy（補正なし）

サンプラーとトレーナーの数値差異を無視してそのまま学習した場合。

- ステップ 318 付近で **KL ダイバージェンスが急激にスパイク**
- それに伴い **損失もスパイク**
- **報酬が途中で崩壊**（reward collapse）

これは on-policy を仮定したアルゴリズムを、実質的に off-policy データで実行したことによる典型的な失敗パターンである。

#### 2. Off-Policy（重要度重み付けあり）

Off-policy 補正として重要度重み付け（importance weighting）を適用した場合。

- KL ダイバージェンスは概ね **0.001 程度**に抑えられ、時折スパイクが発生するものの安定
- 学習は円滑に進行

重要度重み付けにより分布のずれを補正できるが、完全ではなく、散発的なスパイクが残る。

#### 3. True On-Policy（決定的カーネル使用）

Batch invariant カーネルを用いてサンプラーとトレーナーの数値計算を完全に一致させた場合。

- **KL ダイバージェンスが常に 0**（フラット）
- サンプラーとトレーナーのポリシーが完全に一致
- Off-policy 補正が不要で、学習は安定的に進行

### RLVR における非決定性の影響まとめ

| 構成 | KL-divergence | 安定性 | 追加コスト |
|------|--------------|--------|-----------|
| Off-Policy（補正なし） | スパイク発生 | 報酬崩壊のリスク | なし |
| Off-Policy（重要度重み付け） | ~0.001（散発的スパイク） | 概ね安定 | 重要度比の計算コスト |
| True On-Policy（決定的） | 0（完全一致） | 安定 | 推論速度 ~1.6x 低下 |

## 考察

### 非決定性の定義の明確化

本記事で扱った非決定性は、一般的な「GPU の浮動小数点演算の非結合性」とは異なる概念であることを強調したい。

- **Run-to-run determinism**: 同一入力に対して同一出力が得られる性質。個々の GPU カーネルはこれを満たす。
- **Batch invariance**: バッチサイズに依存せず同一出力が得られる性質。多くの推論カーネルはこれを**満たさない**。

推論サーバーにおける非決定性は、batch invariance の欠如とバッチサイズの非決定性（サーバー負荷への依存）の合成によって生じる。

### RLVR 実装への示唆

RLVR の実装において、以下の選択肢が考えられる:

1. **決定的カーネルの採用**: 推論速度を犠牲にして true on-policy を実現する。報酬崩壊のリスクがなく、最もシンプルで信頼性が高い。
2. **重要度重み付けによる補正**: 推論速度を維持しつつ off-policy 補正を行う。追加の実装コストがかかるが、性能面では有利。
3. **サンプラーとトレーナーの統合**: 同一の計算パスでサンプリングと学習を行うことで数値差異を排除する。ただし、推論エンジンの最適化（continuous batching, paged attention 等）を活用できなくなるため、実用的ではない場合が多い。

## まとめ

LLM 推論における非決定性の根本原因は、浮動小数点演算の非結合性それ自体ではなく、**推論カーネルの batch invariance の欠如**と**動的バッチングによるバッチサイズの非決定性**の合成にある。

この問題は推論結果の再現性だけでなく、RLVR において **on-policy 学習を暗黙的に off-policy 化してしまう**という深刻な影響を持つ。決定的な推論カーネルを用いることで、サンプラーとトレーナー間の KL ダイバージェンスを完全にゼロに保ち、true on-policy での安定した学習が可能になる。

---

## References

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) — Thinking Machines
- [thinking-machines-lab/batch-invariant-ops](https://github.com/thinking-machines-lab/batch-invariant-ops) — 参照実装
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [FlexAttention](https://pytorch.org/blog/flexattention/)
