---
title: "Continuous Batching と Prefill-Decode Disaggregation — LLM推論サービングの最適化"
date: 2026-02-08
tags: ["LLM", "Inference", "Serving", "Continuous Batching", "Disaggregation", "NVIDIA Dynamo", "vLLM", "DistServe"]
description: "LLM推論サービングの中核技術である Continuous Batching の仕組みと、Prefill-Decode Disaggregation による更なる最適化を、DistServe (OSDI'24)、NVIDIA Dynamo、Beyond the Buzz (Mitra et al., 2025) の設計・分析を通じて解説する。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

LLM 推論サービングは、単純な forward pass の繰り返しではない。数千〜数万のユーザーリクエストを同時に捌きつつ、**Time to First Token (TTFT)** と **Time Per Output Token (TPOT)** の SLO（Service Level Objective）を満たす必要がある。

本記事では、LLM 推論サービングの2つの中核的な最適化技術を解説する:

1. **Continuous Batching** — リクエストの動的なバッチ管理
2. **Prefill-Decode Disaggregation** — 2つのフェーズを異なる GPU に分離する設計

いずれも NVIDIA Inference Optimization チームの日常的な業務に直結する技術であり、vLLM / SGLang / TensorRT-LLM / NVIDIA Dynamo といった主要な推論フレームワークの根幹をなしている。

## LLM 推論の2つのフェーズ

最適化手法を理解する前に、LLM 推論が本質的に **2つの異なるフェーズ** から構成されていることを押さえる必要がある。

### Prefill フェーズ

ユーザーの入力プロンプト全体を一度に処理し、最初のトークンを生成する。全入力トークンに対する Attention 計算を行い、KV Cache を構築する。

- **計算特性**: Compute-bound（大きな行列積）
- **並列度**: 入力系列長に比例して高い
- **レイテンシ指標**: TTFT（Time to First Token）

### Decode フェーズ

1トークンずつ自己回帰的に生成する。各ステップでは新しいトークン1つに対する Attention 計算のみを行い、KV Cache を逐次更新する。

- **計算特性**: Memory bandwidth-bound（小さな行列-ベクトル積）
- **並列度**: バッチ内のリクエスト数に依存
- **レイテンシ指標**: TPOT（Time Per Output Token）

この2つのフェーズの計算特性の違いが、本記事で解説するすべての最適化の出発点となる。

| | Prefill | Decode |
|---|---|---|
| 計算パターン | 行列-行列積 (GEMM) | 行列-ベクトル積 (GEMV) |
| ボトルネック | Compute-bound | Memory bandwidth-bound |
| GPU 利用率 | 高い（少数リクエストで飽和） | 低い（大バッチが必要） |
| レイテンシ指標 | TTFT | TPOT |
| 最適な並列化戦略 | Tensor Parallelism | Data / Pipeline Parallelism |

## Continuous Batching

### Static Batching の問題

最も単純なバッチ処理は **Static Batching**（Naive Batching）である。複数のリクエストを固定サイズのバッチとしてまとめ、**全リクエストの生成が完了するまでバッチ構成を維持する**。

この方式には致命的な非効率がある。LLM の出力長はリクエストごとに大きく異なるため、短い出力で終了したリクエストの GPU スロットは、最も遅いリクエストが終わるまで**遊休状態**となる。

```
Static Batching のタイムライン:

Req A: [Prefill][Decode][Decode][Decode][EOS][  idle  ][  idle  ][  idle  ]
Req B: [Prefill][Decode][Decode][Decode][Decode][Decode][Decode][EOS]
Req C: [Prefill][Decode][Decode][EOS][  idle  ][  idle  ][  idle  ][  idle  ]
Req D: (waiting...)                                              [Prefill]...
       ──────────────────────── time ────────────────────────────>
```

Req A と Req C が早期に完了しても、Req B の生成が終わるまで新しいリクエストを投入できない。Req D は不必要に長い待ち時間を強いられる。

### Iteration-Level Scheduling

Continuous Batching は **Iteration-Level Scheduling** によってこの問題を解決する。バッチ構成の決定を「リクエスト単位」から「forward pass（イテレーション）単位」に変更し、**各イテレーションで完了したリクエストのスロットに即座に新しいリクエストを挿入する**。

```
Continuous Batching のタイムライン:

Slot 1: [Req A: Prefill][A: Dec][A: Dec][A: EOS][Req D: Prefill][D: Dec]...
Slot 2: [Req B: Prefill][B: Dec][B: Dec][B: Dec][B: Dec][B: Dec][B: EOS]
Slot 3: [Req C: Prefill][C: Dec][C: EOS][Req E: Prefill][E: Dec][E: Dec]...
         ──────────────────────── time ──────────────────────────>
```

Req A が完了した瞬間に Req D が投入され、GPU スロットが常にアクティブなリクエストで埋まる。

### 性能向上の実測値

[Anyscale のベンチマーク](https://www.anyscale.com/blog/continuous-batching-llm-inference) では、Meta OPT-13B（A100 GPU）に対して以下の結果が報告されている:

| 手法 | スループット向上（対 Static Batching） |
|---|---|
| Continuous Batching 単体 | **~8x** |
| Continuous Batching + PagedAttention (vLLM) | **~23x** |

Continuous Batching だけで 8 倍のスループット向上を達成し、[PagedAttention](https://arxiv.org/abs/2309.06180) による KV Cache のメモリ管理最適化を組み合わせることで 23 倍に達する。

### PagedAttention との相乗効果

Continuous Batching の効果を最大化するには、KV Cache のメモリ管理が鍵となる。従来方式では各リクエストに最大系列長分の連続メモリを事前確保するため、**メモリの断片化と無駄が深刻**であった。

[vLLM](https://github.com/vllm-project/vllm) が導入した PagedAttention は、OS の仮想メモリ管理にヒントを得て、KV Cache を**固定サイズのブロック**に分割して動的に割り当てる。これにより:

- メモリ断片化の大幅な削減
- 実効的なバッチサイズの拡大
- Prefix Caching（共通プレフィックスの KV Cache 共有）の実現

が可能になり、Continuous Batching との組み合わせで劇的なスループット向上を実現する。

### Continuous Batching の実装上の課題

Continuous Batching は概念的にはシンプルだが、実装には以下の課題がある:

1. **Prefill の割り込み問題**: 新しいリクエストの Prefill は計算量が大きく、実行中の Decode バッチに割り込むと TPOT が悪化する
2. **スケジューリングポリシー**: Prefill と Decode のどちらを優先するか、Prefill をチャンク分割するかなどの設計判断
3. **メモリ管理**: 動的なバッチ構成変更に伴う KV Cache の割り当て・解放の効率化

特に課題 1 は次章で解説する Prefill-Decode Disaggregation の直接的な動機となる。

## Prefill-Decode Disaggregation

### Colocated Serving の限界

Continuous Batching を採用した現行の推論サーバー（vLLM, SGLang, TensorRT-LLM 等）では、Prefill と Decode が**同一 GPU 上で実行される**（Colocated Serving）。この方式には根本的な問題がある。

#### 問題 1: Prefill-Decode Interference（干渉）

長いプロンプトの Prefill が実行されると、同じ GPU 上で進行中の Decode リクエストが**ブロックされる**。Decode の各ステップは通常数十ミリ秒で完了するが、長い Prefill（数千トークン）は数百ミリ秒を要する。この間、Decode リクエストの TPOT が大幅に悪化する。

```
Colocated Serving での干渉:

GPU: [Dec batch][Dec batch][ Long Prefill (500ms) ][Dec batch][Dec batch]
                           ^^^^^^^^^^^^^^^^^^^^^^^^
                           この間、Decode が停止
                           → TPOT SLO 違反
```

Chunked Prefill（Prefill を小さなチャンクに分割して Decode と交互に実行）はこの問題を緩和するが、完全には解決できない。チャンクが小さすぎると TTFT が増大し、大きすぎると Decode への干渉が残る。

#### 問題 2: 並列化戦略の結合

Colocated Serving では、Prefill と Decode が同じ GPU 群で実行されるため、**並列化戦略を共有せざるを得ない**。しかし前述の通り、両フェーズの最適な並列化戦略は異なる:

- **Prefill**: Compute-bound → 少数のリクエストで GPU を飽和させるため、**Tensor Parallelism (TP)** で単一リクエストの処理を高速化すべき
- **Decode**: Memory bandwidth-bound → 大バッチを効率的に処理するため、**Data Parallelism (DP)** でスループットを稼ぐべき

Colocated Serving ではどちらかに寄せた並列化を選ぶしかなく、片方のフェーズが必ず準最適になる。

#### 問題 3: Goodput の低下

[DistServe (Zhong et al., OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) は、従来のスループット指標の問題を指摘し、**Goodput** という新たな指標を提案した。

$$\text{Goodput} = \text{SLO を満たすリクエストの最大処理レート (requests/sec)}$$

例えば、システムが 10 req/s のスループットを達成していても、SLO を満たすのが 3 件だけなら Goodput は 3 req/s に過ぎない。Colocated Serving では Prefill-Decode 干渉により、**スループットが高くても Goodput が低い**という状況が生じる。

### DistServe: Disaggregation の原論文

[DistServe](https://arxiv.org/abs/2401.09670) は、Prefill と Decode を**異なる GPU プールに分離**することで上記の問題を解決する。

#### アーキテクチャ

```
                    ┌──────────────────────┐
                    │   Request Router     │
                    └──────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐    ┌─────────▼─────────┐
    │  Prefill GPU Pool  │    │  Decode GPU Pool   │
    │                    │    │                    │
    │  - TP for低レイテンシ │    │  - DP for高スループット│
    │  - 少数リクエスト     │    │  - 大バッチ処理      │
    │  - TTFT 最適化      │    │  - TPOT 最適化      │
    └────────┬───────────┘    └────────▲───────────┘
             │                         │
             └──── KV Cache Transfer ──┘
```

処理の流れ:

1. **Request Router** がリクエストを Prefill GPU Pool に割り当て
2. **Prefill GPU** が入力プロンプトを処理し、KV Cache を生成
3. **KV Cache** を Decode GPU に転送
4. **Decode GPU** が自己回帰的にトークンを生成

#### KV Cache 転送のオーバーヘッド

Disaggregation の最大の懸念は KV Cache 転送のコストである。DistServe の分析では、OPT-175B・2048 トークンのリクエストに対して:

| インターコネクト | 転送レイテンシ | Decode 1ステップ | オーバーヘッド比 |
|---|---|---|---|
| PCIe 5.0 | ~17.6 ms | 30-50 ms | 35-59% |
| NVLink (600 GB/s) | 大幅に削減 | 30-50 ms | ≪ 1 step |

NVLink 環境では KV Cache 転送を Decode の計算と**オーバーラップ**させることで、実質的なオーバーヘッドを最小化できる。

#### 性能結果

DistServe は vLLM（Colocated Serving）と比較して、以下の Goodput 向上を達成した:

| ワークロード | Goodput 向上 | 備考 |
|---|---|---|
| Chatbot | 2.0-3.41x | 中程度の入出力長 |
| Code Completion | 3.2x | 短い入力、短い出力 |
| Summarization | **4.48x** | 長い入力、短い出力 |

Summarization の向上が最大であるのは、長い入力の Prefill が Decode に与える干渉が最も深刻なワークロードだからである。

### Chunked Prefill との比較

Disaggregation の代替手法として **Chunked Prefill**（Dynamic SplitFuse）がある。これは Prefill を小さなチャンクに分割し、Decode バッチに「相乗り」させる方式である。

| | Disaggregation | Chunked Prefill |
|---|---|---|
| 干渉の除去 | 完全 | 部分的（チャンクサイズ依存） |
| 追加コスト | KV Cache 転送 | なし |
| 並列化の独立最適化 | 可能 | 不可能 |
| 実装の複雑さ | 高い | 低い |
| GPU 追加投資 | 必要（別プール） | 不要 |

Chunked Prefill はチャンクサイズに関するトレードオフを内包する:

- **チャンクが小さすぎる**: TTFT が線形に増大（多数のイテレーションが必要）
- **チャンクが大きすぎる**: Decode への干渉が残り、KV Cache の再ロードコストが二次的に増大

Disaggregation はこれらのトレードオフを根本的に解消するが、KV Cache 転送と GPU プール管理の複雑さを引き受ける。**負荷が高いシナリオでは Disaggregation が圧倒的に有利**であり、低負荷では Colocated + Chunked Prefill で十分な場合もある。

## NVIDIA Dynamo: プロダクション Disaggregation

[NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/design_docs/disagg_serving.html) は、DistServe の概念をプロダクション品質で実装した推論フレームワークであり、Disaggregated Serving を実運用レベルで提供する。

### アーキテクチャコンポーネント

Dynamo の Disaggregated Serving は以下の4つのコンポーネントから構成される:

| コンポーネント | 役割 |
|---|---|
| **Worker** | Prefill・Decode 両方を実行可能な汎用ワーカー |
| **Prefill Worker** | Prefill 専用のワーカー |
| **Disaggregated Router** | Prefill をローカル実行するかリモート（専用 Prefill Worker）に送るかを判断 |
| **Prefill Queue** | NATS ストリームによるグローバルなリクエストキュー。Prefill Worker 間の負荷分散を実現 |

### Conditional Disaggregation

Dynamo の重要な設計判断は、**すべてのリクエストを disaggregate するのではなく、条件付きで判断する**点にある。以下の2条件を**両方**満たす場合のみリモート Prefill を実行する:

1. **Prefix Cache ヒット後の実効 Prefill 長が閾値を超える**: 短い Prefill はローカル実行の方が KV Cache 転送コストを回避できて効率的
2. **Prefill Queue のリクエスト数が閾値未満**: キューが溢れている場合はリモート Prefill に送っても待ち時間が増えるだけ

この Conditional Disaggregation により、ワークロードの特性に応じて**動的に Colocated と Disaggregated を切り替える**。

### KV Cache 転送: NIXL

Dynamo は **NIXL**（NVIDIA Inter-node eXchange Library）を用いて、GPU VRAM 間の**非同期直接転送**を実現する。

主要な最適化:

- **Lazy Metadata Loading**: メモリディスクリプタを ETCD にキャッシュし、リクエストごとのオーバーヘッドをブロック ID の送信のみに削減
- **Block Consolidation**: 連続する KV ブロックを結合し、ディスクリプタ数を最小化
- **Layout Transposition**: Prefill と Decode で異なる TP 構成を使う場合、高性能カーネルで KV Cache のレイアウト変換を実行

### 負荷レベルに応じた戦略

Dynamo のドキュメントは、負荷レベルに応じた最適戦略を明示している:

| 負荷 | 推奨構成 | 理由 |
|---|---|---|
| **低負荷** | Monolithic (Colocated) | Disaggregation のオーバーヘッドが利点を上回る |
| **中負荷** | Disaggregated | ITL と TTFT の両方が改善 |
| **高負荷** | Disaggregated + Prefill Engine 最小化 | Decode 側の KV Cache 容量を最大化 |

### Prefill / Decode エンジンのチューニング

#### Prefill エンジン

- **目標**: GPU を飽和させる最小バッチサイズで運用し、TTFT を最小化
- 入力系列長が ~1000 トークン未満では GPU が飽和しないため効率が低下
- 現行の Dynamo ではバッチサイズ 1 に制限されており、`max-local-prefill-length` で閾値を調整

#### Decode エンジン

- **目標**: バッチサイズを最大化しつつ KV Cache の容量を確保
- ブロックサイズは Dense モデルで 128 が推奨（断片化と Prefix Cache ヒット率のバランス）
- バッチサイズを増やすと中間テンソルのメモリが増加し、KV Cache 容量が減少するトレードオフ

### 並列化戦略

Dense モデルに対する推奨:

> TP within node, PP across nodes

例: Llama-405B w8a8 (H100)
- 単一ノード: TP8
- 2ノード: TP8 × PP2

TP を増やすことで単一リクエストのレイテンシは改善するが、通信オーバーヘッドにより GPU あたりのスループットは低下する。Llama-3.3-70B NVFP4 (B200) では TP1→TP4 で GPU あたり約 1.28 倍の改善に留まる。

## GB200 NVL72 と MoE モデルの推論最適化

[NVIDIA GB200 NVL72](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/) は、Disaggregated Serving と MoE モデルの推論を前提に設計された次世代プラットフォームである。

### ハードウェアの進化

| | HGX H200 | GB200 NVL72 |
|---|---|---|
| GPU 数 | 8 | 最大 72 |
| GPU 間帯域幅 | 900 GB/s/GPU | **1.8 TB/s/GPU** |
| All-to-All 帯域幅 | — | **130 TB/s（集約）** |
| 対 400G Ethernet | — | **36x** |

### MoE モデルにおける Expert Parallelism

Mixture-of-Experts (MoE) モデルでは、各トークンが選択されたエキスパートにルーティングされ、計算後に All-to-All 通信で結果を交換する。

DeepSeek R1（256 エキスパート）の Decode フェーズでは、**Wide Expert Parallelism** が有効である。各 GPU が少数のエキスパート（256/64 = 4 エキスパート/GPU）のみを保持し、64 GPU で分散処理する。この場合、All-to-All 通信が性能のボトルネックとなるため、NVLink の 1.8 TB/s/GPU が決定的に重要となる。

### Disaggregation × MoE の相乗効果

MoE モデルでは Prefill と Decode の計算特性の差がさらに顕著であり、Disaggregation の効果が大きい:

| モデル種別 | Disaggregation によるスループット向上 |
|---|---|
| Dense (Llama 70B) | 最大 **3x** |
| MoE (DeepSeek R1) | 最大 **6x** |

### Dynamo Planner

Dynamo には **Planner** と呼ばれる動的リソース管理コンポーネントがある。以下の指標を監視し、Prefill/Decode リソースの配分を自動調整する:

- Prefill Queue の待ち時間
- Decode 側の KV Cache メモリ使用率
- アプリケーションの SLO 要件

入出力系列長の変動に応じて、Aggregated Serving（Colocated）と Disaggregated Serving を動的に切り替える。

## Disaggregation は万能か？— 大規模デザインスペース探索

[Beyond the Buzz: A Pragmatic Take on Inference Disaggregation (Mitra et al., 2025)](https://arxiv.org/abs/2506.05508) は、NVIDIA による**初の大規模かつ体系的な Disaggregated Inference の実証分析**である。数十万のデザインポイントを評価し、Disaggregation が真に有効な条件とそうでない条件を定量的に明らかにした。

### Disaggregation が有効な場合と有効でない場合

この論文の核心的なメッセージは、**Disaggregation は万能ではなく、ワークロード特性・モデルサイズ・ハードウェア構成の組み合わせによって効果が大きく変動する**ということである。

#### Disaggregation が有効なケース

- **Prefill-heavy なトラフィック**（入力系列長 >> 出力系列長）: Summarization や RAG のように長いコンテキストを処理するワークロード
- **大規模モデル**（>10B パラメータ）: 並列化戦略の探索空間が広く、Prefill と Decode で異なる最適構成を選択できる余地が大きい
- **緩和されたレイテンシ制約**: スループット最大化を目指す場合に効果が顕著

#### Disaggregation の効果が限定的なケース

- **Generation-heavy なトラフィック**（出力系列長が長い）: Decode が支配的なワークロードでは、分離のオーバーヘッドが利点を上回る
- **小規模モデル**（<10B パラメータ）: 並列化の選択肢が限られ、Colocated Serving と比較して明確な優位性が出にくい
- **厳しい TTFT 制約下**: FTL（First Token Latency）が厳しい場合、KV Cache 転送のオーバーヘッドが無視できなくなる

この知見は、前節の Dynamo における Conditional Disaggregation（負荷レベルに応じた動的切替）の設計判断を裏付けている。

### モデルサイズの影響

Llama-3.1 シリーズ（8B, 70B, 405B）に対する Pareto 分析では、モデルサイズが大きくなるほど Disaggregation の利点が増大することが示された。これは大規模モデルほど TP / EP / PP の組み合わせが豊富であり、Prefill と Decode で**異なる最適な並列化戦略**を選択できるためである。

### Chunked Pipeline Parallelism (CPP)

長コンテキスト処理の FTL 制約を満たすため、論文は **Chunked Pipeline Parallelism (CPP)** を提案している。入力系列を小さなチャンクに分割し、前のチャンクの KV Cache を利用しつつ、パイプライン並列性で異なるチャンクの異なる層を重複実行する。

```
CPP の動作イメージ（4チャンク × 4パイプラインステージ）:

Time →
Stage 1: [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
Stage 2:          [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
Stage 3:                   [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
Stage 4:                            [Chunk 1][Chunk 2][Chunk 3][Chunk 4]
```

DeepSeek-R1 で 256K トークンを 64 GPU（EP×PP=64）で処理する場合、CPP により高スループットを維持しつつ FTL を大幅に短縮できることが示された。これは過度な Tensor Parallelism に頼らずに長系列を処理する手法として重要である。

### Piggybacking と MLA のオーバーヘッド

**Piggybacking** は Colocated Serving の改良手法であり、新しいリクエストの Prefill をチャンク化して既存の Decode バッチに相乗りさせることで、Decode の停止を軽減する。Generation-heavy なトラフィックや緩いレイテンシ制約下で有効である。

ただし、DeepSeek-R1 の **Multi-Latent Attention (MLA)** では、各 Prefill チャンクで down/up projection の冗長な再計算が必要となるオーバーヘッドが発生する。これは前のチャンクの up-projected KV 値を一時キャッシュすることで緩和されるが、MLA アーキテクチャ固有の課題として認識されている。

### 動的 Rate Matching の重要性

Disaggregated Serving において、Prefill GPU と Decode GPU の比率（Rate Matching）は性能に決定的な影響を与える。論文は、**固定比率による運用が大幅な性能劣化を引き起こす**ことを実証した。

例えば、Prefill:Decode の GPU 比率を 3.5 に固定すると、緩いレイテンシ制約では最適に近い性能を発揮するが、厳しいレイテンシ制約では大幅に劣化する。逆に 0.5 の比率は厳しい制約下で有効だが、高スループット領域では不十分となる。

$$\alpha = \frac{\text{Best Prefill Throughput per GPU}}{\text{Decode Request Throughput per GPU}}$$

この最適比率 $\alpha$ はモデル・ワークロード・レイテンシ制約の組み合わせによって 0.5〜3.5+ まで大きく変動するため、**動的な GPU 配分調整機構**（Dynamo Planner のような）が不可欠である。

### KV Cache 転送帯域幅の分析

論文は、KV Cache 転送の帯域幅要件を定式化し、現行のデータセンターインフラで十分であることを示した。

**Egress 帯域幅**（Prefill → Decode）:

$$BW_{\text{egress}} = \frac{L \times B_{\text{prefill}} \times \text{ISL} \times d_{\text{head}} \times n_{\text{kv}} \times \text{bytes}}{FTL \times N_{\text{prefill}}}$$

重要な知見として、**ISL が増加すると Egress 帯域幅の要件は減少する**。これは FTL が Attention の二次的な計算コストにより ISL に対して超線形にスケールする一方、KV Cache サイズは線形にしかスケールしないためである。

**Ingress 帯域幅**（Decode が受信）:

$$BW_{\text{ingress}} = \frac{L \times B_{\text{decode}} \times \text{ISL} \times d_{\text{head}} \times n_{\text{kv}} \times \text{bytes}}{TTL \times \text{OSL} \times N_{\text{decode}}}$$

Ingress 帯域幅は **OSL に反比例**する。出力が長いほど TTL が増加し、per-GPU の帯域幅要件が低下する。NVLink 環境ではいずれのケースでもボトルネックとならないことが確認されている。

### NVLink ドメインサイズの影響

NVLink ドメイン（高帯域幅で接続された GPU 群のサイズ）が大きいほど、Disaggregation の性能が向上する。これは、大きなドメインが Expert Parallelism や Tensor Parallelism の選択肢を広げ、Prefill・Decode それぞれで最適な構成を選択する自由度を高めるためである。DeepSeek-R1 では特に中程度のレイテンシ制約下で、より高い EP と大きなバッチサイズの採用が可能となった。

### 実務への示唆

論文は以下のデプロイメント指針を提示している:

1. **自身のワークロードの ISL/OSL 分布を把握**してから Disaggregation の採否を判断する
2. **動的 Rate Matching を実装**し、固定比率による性能劣化を回避する
3. **大きな NVLink ドメイン**を活用して並列化の柔軟性を最大化する
4. **長コンテキスト処理には CPP** を検討し、TP 過剰に頼らない設計とする
5. **トラフィック特性の変動を監視**し、ISL/OSL 分布の変化に応じて構成を動的に調整する

## Disaggregated Inference の現在地 — 18ヶ月後の振り返り

[Hao AI Lab の振り返り記事](https://hao-ai-lab.github.io/blogs/distserve-retro/) は、DistServe 発表から約18ヶ月後の Disaggregation エコシステムの現状を総括している。この節では、論文の理論から実運用への移行で見えてきた知見を整理する。

### なぜ 2025 年に採用が爆発的に広がったか

DistServe の発表（2024年1月）後、2024年中は採用が限定的であった。既存の推論サーバーの大規模リファクタリングが必要であり、エンジニアリングコストが障壁となっていた。しかし 2025 年に入り、以下の3つの要因で状況が急変した:

1. **ビジネスクリティカル化**: LLM がアプリケーションのコアコンポーネントとなり、レイテンシ制御がスループット以上に事業の成長（場合によっては存続）に直結するようになった
2. **スケールの要請**: モデルサイズとトラフィックの増大により、数百〜数千 GPU 規模のクラスタ運用が常態化。Disaggregation によるフェーズ独立のリソース配分が真に威力を発揮するスケールに到達した
3. **アーキテクチャの構成可能性**: Disaggregation が推論スタック全体（ハードウェア、システム、ストレージ）にわたる最適化の「接合面」を提供し、個別コンポーネントの独立進化を可能にした

### プロダクションエコシステムの三層構造

2025年時点で、Disaggregated Serving のエコシステムは以下の三層に整理できる:

#### オーケストレーション層

| フレームワーク | 特徴 |
|---|---|
| **NVIDIA Dynamo** | Prefill/Decode を first-class citizen として扱い、KV-aware routing を実装。GB200 NVL72 で SOTA 達成。NIXL による統一的な転送抽象化（NVLink, InfiniBand, PCIe, SSD） |
| **llm-d** | Kubernetes ネイティブの Disaggregation。厳密な SLO 制御と異種クラスタ間の弾力的スケーリング |
| **Ray Serve LLM** | Ray ベースのモジュラー設計。NIXL/LMCache コネクタと KV-affinity routing を提供 |

#### ストレージ層

| フレームワーク | 特徴 |
|---|---|
| **LMCache** | KV Cache を推論エンジンから分離。バッチ化データ移動と I/O パイプラインにより転送を最適化 |
| **MoonCake** (FAST'25 Best Paper) | 未使用ストレージをプールし、クラスタ全体の集中的な KV Cache 抽象化を提供。任意の Prefill ノードから任意の Decode ノードへの KV Cache ハンドオフを実現 |

#### エンジン層

| エンジン | 実測性能 |
|---|---|
| **SGLang** | DeepSeek R1 (96 H100, Prefill 3ノード + Decode 9ノード): 52.3k input TPS, 22.3k output TPS/ノード。GB200 では Prefill 3.8x, Decode 4.8x の向上 |
| **vLLM / llm-d 0.3** | DeepSeek R1: 2.2k TPS (32-way EP) / 2.0k TPS (96-way EP) per H200 GPU |
| **TensorRT-LLM** | DeepSeek R1 で 60k TPS 超。GPT-OSS 120B で 1000 TPS |

### DeepSeek のデプロイメント構成

DeepSeek R1 の実運用構成は、Disaggregation のリファレンスアーキテクチャとして参考になる:

- **Prefill**: 3 ノード（各 8 H100）— 小さめの EP/DP で大規模プロンプトを処理
- **Decode**: 9 ノード（各 8 H100）— Wide EP (~256) + 高 DP (8-16) で GroupGEMM の利用率を最大化
- **ストレージ**: 3FS ライブラリが SSD スループットとネットワーク帯域幅を統合し、ロカリティを意識しないストレージアクセスを実現

Prefill : Decode = 1 : 3 の比率は、Decode フェーズが自己回帰的に多くのステップを要するためである。この比率はワークロード（入出力長の分布）によって動的に調整されるべきものであり、Dynamo Planner のような自動調整機構の重要性を示している。

### ハードウェアの専門分化

Disaggregation は、Prefill と Decode に**異なるハードウェア**を割り当てることを可能にする。これは以下の方向で進展している:

- **コスト最適化**: Prefill には計算能力の高い GPU、Decode にはメモリ帯域幅の広い GPU を選択し、TCO を削減
- **ハードウェア協調設計**: ベンダーが Attention パスと FFN パスを独立に最適化
- **専用 ASIC の開発**: Huawei (Ascend NPU), Enflame, MetaX, Biren 等が Decode 特化・Attention 最適化 ASIC を試作・実戦投入中
- **NVIDIA Rubin CPX**: 長コンテキスト推論向けに Prefill-Decode Disaggregation を前提として設計された次世代アーキテクチャ

### 次のフロンティア: Attention-FFN Disaggregation

Prefill-Decode Disaggregation のさらに先に、**Attention と FFN の分離**（Attention-FFN Disaggregation, AFD）がある。

Attention は memory-bound、FFN は compute-bound という性質の違いに着目し、これらを異なる計算リソースに割り当てる。従来は Activation の転送オーバーヘッドが障壁とされてきたが、**MoE モデルでは状況が異なる**。

MoE の Expert Parallelism では、各 Decode ステップで既に2回の All-to-All 通信が発生する。Attention-FFN の分割をこの既存の通信パターンに整合させることで、AFD のオーバーヘッドは**実質的にゼロ**となる。MegaScale-Infer や Stepfun が DeepSeek R1・Qwen3-235B でこのアプローチの実現可能性を実証した。

一方、Dense モデルでは通信オーバーヘッドが依然として課題であり、計算と通信のオーバーラップが困難な状況が続いている。

### 業界への波及効果

DistServe が提唱した **TTFT** と **TPOT** は、発表から18ヶ月で推論ベンチマークにおける**標準的なレイテンシ指標**として定着した。これにより、業界全体の関心が raw throughput から**レイテンシ SLO 制御**にシフトした。

Disaggregation を起点として、以下の研究領域が急速に拡大している:

- **KV Cache 最適化**: CacheGen, MemServe
- **並列化構成探索**: [Beyond the Buzz (Mitra et al.)](https://arxiv.org/abs/2506.05508)
- **スケジューリング最適化**: SLO-Serve
- **ハイブリッド Disaggregation**: TaiChi（Disaggregated と Aggregated の動的切替）
- **マルチモーダル対応**: ModServe
- **異種ハードウェア活用**: Helix, HexGen-2, CENT
- **ネットワーク最適化**: FuseLink
- **強化学習適応**: StreamRL
- **電力効率**: EcoServe, GreenLLM

## まとめ

本記事で解説した技術の関係を整理する:

```
Static Batching
    │
    │ GPU スロットの遊休を解消
    ▼
Continuous Batching (+ PagedAttention)
    │
    │ Prefill-Decode 干渉を解消
    ▼
Prefill-Decode Disaggregation (DistServe, 2024)
    │
    │ プロダクション実装 + Conditional Disaggregation
    ▼
NVIDIA Dynamo / llm-d / Ray Serve LLM (2025)
    │
    │ MoE + Expert Parallelism + NVLink
    ▼
GB200 NVL72 / Rubin CPX 最適化
    │
    │ Attention と FFN の分離
    ▼
Attention-FFN Disaggregation (次世代)
```

| 技術 | 解決する課題 | 核心的なアイデア |
|---|---|---|
| Continuous Batching | GPU スロットの遊休 | Iteration-level scheduling |
| PagedAttention | KV Cache のメモリ断片化 | ブロック単位の動的メモリ管理 |
| P/D Disaggregation | Prefill-Decode 干渉、並列化の結合 | フェーズごとの GPU プール分離 |
| Conditional Disaggregation | 低負荷時のオーバーヘッド | 動的な Colocated/Disaggregated 切替 |
| NIXL / LMCache / MoonCake | KV Cache 転送・管理コスト | 非同期 GPU-to-GPU 転送 + 集中ストレージ抽象化 |
| Expert Parallelism | MoE の通信ボトルネック | Wide EP + 高帯域 NVLink |
| Dynamic Rate Matching | 固定 GPU 比率の性能劣化 | ワークロードに応じた Prefill/Decode GPU 配分の動的調整 |
| Chunked Pipeline Parallelism | 長コンテキストの FTL 制約 | 入力系列のチャンク分割 + パイプライン重複実行 |
| Attention-FFN Disaggregation | Attention/FFN の計算特性の不一致 | MoE の既存 All-to-All に整合した分離 |

NVIDIA Inference Optimization チームでのインターンにおいては、これらの技術が個別の最適化ではなく、**推論パイプライン全体の設計空間**を構成していることを意識することが重要である。あるコンポーネントの変更が他のコンポーネントの最適なパラメータを変える（例: Disaggregation の導入により各フェーズの最適な TP/PP 構成が変わる）ため、**システム全体を俯瞰した最適化**が求められる。

さらに、DistServe の振り返りが示すように、この分野は論文発表から18ヶ月で研究段階からプロダクション必須技術へと急速に成熟した。ハードウェア（Rubin CPX, Decode 特化 ASIC）、システム（Dynamo, llm-d）、ストレージ（MoonCake, 3FS）の三層が co-evolve しており、いずれの層にも最適化の余地が残されている。

## 参考文献

- [Continuous Batching for LLM Inference (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Zhong et al., OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [NVIDIA Dynamo — Disaggregated Serving](https://docs.nvidia.com/dynamo/latest/design_docs/disagg_serving.html)
- [NVIDIA Dynamo — Performance Tuning](https://docs.nvidia.com/dynamo/latest/performance/tuning.html)
- [How NVIDIA GB200 NVL72 and NVIDIA Dynamo Boost Inference Performance for MoE Models](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP'23)](https://arxiv.org/abs/2309.06180)
- [DistServe Blog (Hao AI Lab)](https://hao-ai-lab.github.io/blogs/distserve/)
- [Disaggregated Inference: 18 Months Later (Hao AI Lab)](https://hao-ai-lab.github.io/blogs/distserve-retro/)
- [MoonCake: A KV Cache-Centric Disaggregated Architecture for LLM Serving (FAST'25)](https://www.usenix.org/conference/fast25/presentation/qin-ruoyu)
- [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation (Mitra et al., 2025)](https://arxiv.org/abs/2506.05508)
