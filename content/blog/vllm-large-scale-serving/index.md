---
title: "vLLM Large Scale Serving 完全解説 — DeepSeek を H200 で 2.2k tok/s/GPU に到達させた技術スタック"
date: 2026-02-08
tags: ["LLM", "Inference", "vLLM", "DeepSeek", "MoE", "Expert Parallelism", "Disaggregated Serving"]
description: "vLLM の Large Scale Serving を、PagedAttention の基礎から Wide-EP・Dual Batch Overlap・P/D Disaggregation まで体系的に解説する。"
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
draft: true
---

> 本記事は [vLLM Blog: Large Scale Serving](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)、[Aleksa Gordic の vLLM 解説](https://www.aleksagordic.com/blog/vllm)、および [LMSYS の 96 H100 デプロイ記事](https://lmsys.org/blog/2025-05-05-large-scale-ep/) の内容を統合し、NVIDIA Inference Optimization Research Intern を目指すエンジニアに向けて体系的にまとめたものである。

## はじめに

vLLM は 2023 年に PagedAttention を提案した推論フレームワークとして登場し、現在では Meta、LinkedIn、Red Hat、Mistral、HuggingFace など多くの企業が本番環境で利用している。2025 年 12 月、vLLM チームは **Coreweave の H200 クラスタ上で DeepSeek-R1 を 2.2k tokens/sec/GPU** で動作させたことを発表した。これは従来の約 1.5k tok/s/GPU から大幅な進歩であり、その裏には Wide-EP、Dual Batch Overlap (DBO)、Expert Parallel Load Balancing (EPLB)、Prefill/Decode Disaggregation (P/D) といった一連の技術革新がある。

本記事では、vLLM のコアエンジンから大規模サービングまでを**ボトムアップに**解説する。

---

## 1. vLLM コアエンジンの基礎

### 1.1 PagedAttention と KV キャッシュ管理

LLM 推論において、KV キャッシュはメモリ消費の最大のボトルネックである。従来の実装ではリクエストごとに連続したメモリ領域を確保する必要があり、最大系列長分のメモリを事前に予約せざるを得なかった。これにより、実際には使われないメモリが大量に生じ、同時に処理できるリクエスト数（スループット）が低下する。

vLLM の **PagedAttention** は、OS の仮想メモリにおけるページングの概念を KV キャッシュに適用する。KV キャッシュを固定サイズの **ブロック** に分割し、必要に応じてブロック単位で割り当てる。

各ブロックのサイズは以下で計算される:

$$\text{block\_bytes} = 2 \times B_s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}$$

ここで $B_s$ はブロックサイズ（デフォルト 16 トークン）、$2$ は Key と Value の 2 つのテンソルに対応する。

KV キャッシュマネージャは以下のデータ構造を管理する:

- **`free_block_queue`**: 利用可能なブロックのプール（数十万ブロック）
- **`req_to_blocks`**: リクエスト ID → 割り当て済みブロックのマッピング
- **`cached_block_hash_to_block`**: ハッシュ → ブロックのマッピング（Prefix Caching 用）

新しいトークンが生成されると、`allocate_slots` 関数が必要ブロック数 $\lceil n_{\text{new}} / B_s \rceil$ を計算し、プールから確保する。プールが枯渇した場合は低優先度リクエストを **preemption**（退避）して空きを作る。

### 1.2 Continuous Batching

従来の static batching では、バッチ内の全リクエストが終了するまで新しいリクエストを受け付けられなかった。vLLM の **continuous batching** は、各ステップでリクエストの追加・完了を動的に処理する。

実装上のポイントは、全リクエストのトークン列を**1 つの「スーパーシーケンス」にフラット化**し、カスタムの paged attention カーネルで処理する点にある。位置インデックスとアテンションマスクにより、各シーケンスは自身のトークンにのみ attend する。これにより right-padding が不要になり、GPU 演算効率が大幅に向上する。

### 1.3 スケジューリング

エンジンの各ステップは **schedule → forward pass → postprocess** の 3 段階で構成される。

スケジューラは 2 つのキューを管理する:

1. **`running` キュー**: デコード中のリクエスト（最優先で処理）
2. **`waiting` キュー**: プリフィル待ちの新規リクエスト（FCFS またはプライオリティベース）

各ステップで、まず `running` キュー内のリクエストに新トークン分のスロットを確保し、残りのトークンバジェットで `waiting` キューからリクエストをプリフィルに投入する。

### 1.4 Chunked Prefill

長いプロンプト（数千〜数万トークン）を一度にプリフィルすると、そのステップ全体がそのリクエストに独占され、他のデコードリクエストの ITL（Inter-Token Latency）が急増する。

**Chunked Prefill** はプロンプトを `long_prefill_token_threshold` 以下のチャンクに分割し、複数ステップにわたってプリフィルする。これにより prefill と decode のリクエストを同一ステップ内で混在させ、レイテンシの急増を抑制する。

### 1.5 Prefix Caching

同一のシステムプロンプトや Few-shot の prefix を持つリクエストが多い場合、毎回プリフィルを再計算するのは無駄である。

vLLM の **Prefix Caching** は KV ブロックをハッシュベースでキャッシュする:

1. リクエストのトークン列を 16 トークン単位のチャンクに分割
2. 各チャンクのハッシュを計算（前ブロックのハッシュ + 現トークン列 + メタデータ）
3. `find_longest_cache_hit` でキャッシュ済みブロックを検索
4. ヒットした場合はブロックを再利用し、差分のみプリフィル

参照カウントによりブロックの生存管理を行い、カウントが 0 になったブロックは `free_block_queue` に戻る。再割り当て時にハッシュとの不整合が検出された場合はキャッシュエントリを無効化する。

### 1.6 実行モード: Eager vs CUDA Graph

vLLM は 2 つの実行モードをサポートする:

- **Eager モード**: 標準の PyTorch forward pass。デバッグや `torch.compile` との併用に有用
- **Captured モード**: 初期化時に CUDA Graph として forward pass の DAG を記録し、ランタイムでは replay するだけ。カーネル起動のオーバーヘッドを大幅に削減

ウォームアップ時に代表的なバッチサイズで CUDA Graph をキャプチャし、推論時は最も近いサイズのグラフを選択して実行する。`--enforce-eager` フラグで CUDA Graph を無効化できる。

---

## 2. Speculative Decoding

LLM のデコードフェーズは **メモリ帯域律速** であり、1 トークンずつ逐次生成する。Speculative Decoding は、軽量な「ドラフトモデル」で $k$ トークンを高速に提案し、ターゲットモデルで一括検証することで、1 ステップで複数トークンを生成する手法である。

### 2.1 アルゴリズム

1. ドラフトモデルが $k$ 個の候補トークンを生成
2. ターゲットモデルが (コンテキスト + $k$ 候補) に対して forward pass を実行し、$k+1$ 個の確率分布を得る
3. 左から順に検証: $p_{\text{target}}(t) \geq p_{\text{draft}}(t)$ なら採択、そうでなければ確率 $p_{\text{target}} / p_{\text{draft}}$ で採択
4. 最初の棄却で停止。全て採択された場合は $(k+1)$ 番目のトークンを「無料で」サンプル

棄却時のリバランス分布は $\max(0, p_{\text{target}} - p_{\text{draft}})$ を正規化して使用する。

### 2.2 vLLM での実装方式

| 方式 | 概要 |
|------|------|
| **n-gram** | 過去の系列中で現在のサフィックスと一致するパターンを検索し、その後続 $k$ トークンを提案 |
| **EAGLE** | Transformer スタックを軽量 MLP に置換。埋め込み層と LM Head はターゲットモデルと共有 |
| **Medusa** | ターゲットモデルの隠れ状態の上に補助的な線形ヘッドを追加学習し、$k$ トークンを並列予測 |

---

## 3. 性能分析の基礎: Roofline Model

大規模サービングの最適化を議論する前に、推論の性能特性を理解する必要がある。

### 3.1 主要メトリクス

| メトリクス | 定義 |
|-----------|------|
| **TTFT** (Time to First Token) | リクエスト送信から最初のトークン生成までの時間 |
| **ITL** (Inter-Token Latency) | 連続するトークン間の生成間隔 |
| **TPOT** (Time Per Output Token) | ITL の平均 |
| **E2E Latency** | TTFT + $\sum$ ITL |
| **Throughput** | tokens/sec または requests/sec |
| **Goodput** | SLO を満たすスループット |

### 3.2 Roofline 解析

推論の 1 ステップにおいて、重み読み出しの I/O（帯域律速）と行列演算（演算律速）のどちらがボトルネックになるかを判定するのが **Roofline Model** である。

$$\text{Operational Intensity} = \frac{\text{FLOPs}}{\text{Bytes}} \quad [\text{FLOP/Byte}]$$

- **バッチサイズ $B < B_{\text{sat}}$**: 重みの読み出しが支配的（帯域律速）。ステップ時間はほぼ一定で、$B$ を増やしてもレイテンシはほとんど増えない → スループットが線形に向上
- **バッチサイズ $B > B_{\text{sat}}$**: 演算が支配的（演算律速）。$B$ の増加に伴いステップ時間が線形に増加

$$B_{\text{sat}} = \frac{\text{Peak TFLOPS}}{\text{HBM Bandwidth (TB/s)}}$$

H200 (141 GB HBM3e) の場合、ピーク FP16 性能 989 TFLOPS、帯域幅 4.8 TB/s より $B_{\text{sat}} \approx 206$ となる。つまり、デコードフェーズではバッチサイズ 200 程度まではほぼ「無料で」リクエストを追加でき、それ以上で演算律速に遷移する。

---

## 4. DeepSeek アーキテクチャと大規模サービングの課題

### 4.1 DeepSeek-V3/R1 の特徴

DeepSeek-R1 は **671B パラメータの MoE モデル** だが、各 forward pass で活性化されるのは **37B パラメータのみ** である。主要な特徴は以下の通り:

- **Multi-head Latent Attention (MLA)**: KV を低次元の潜在空間に射影することで KV キャッシュのメモリ消費を大幅に削減
- **Mixture of Experts (MoE)**: 256 の routed expert + shared expert で構成。各トークンは上位 8 experts に routing される
- **SwiGLU FFN**: 各 expert の FFN に SwiGLU 活性化を使用

### 4.2 Tensor Parallelism の限界

従来の **Tensor Parallelism (TP)** で DeepSeek を配置すると、以下の問題が発生する:

1. **MLA の潜在射影が全 TP ランクに複製**される → KV キャッシュの有効メモリが TP 度数分の 1 に低下
2. **MoE の expert が各ランクに分散**されるが、疎な活性化パターンにより負荷不均衡が発生
3. **Dense FFN の中間次元 18,432** は高い TP 度数で分割すると GPU のアライメント境界（128）に合わない非効率なセグメントが生じる

実測では TP16 配置の H200 で **34 GB の空きメモリ** しか残らず、大きなバッチサイズを取れない。

### 4.3 Data Parallelism の優位性

Dense FFN 層において TP より DP を優先すべき理由:

- TP は AllReduce（reduce-scatter + all-gather の 2 回通信）を必要とする
- DP は 1 回の reduce-scatter + all-gather で済み、**通信量が 50% 削減**される
- 最適な TP サイズは以下で概算可能:

$$\text{TP}_{\text{opt}} = \sqrt{\frac{N_{\text{param}}}{(1+k) \times N_{\text{hidden}}}}$$

---

## 5. Wide Expert Parallelism (Wide-EP)

### 5.1 概要

**Wide-EP** は Expert Parallelism (EP) と Data Parallelism (DP) を組み合わせた並列化戦略で、DeepSeek のような大規模 MoE モデルの推論に最適化されている。

核心的なアイデアは以下の通り:

- **Expert 集合をデプロイメント全体のランクで共有**する（各ランクが全 expert を持つのではなく）
- **Attention 層は各ランクで独立に実行**（DP Attention）
- トークンは expert 処理のためにランク間を **all-to-all 通信** で移動

### 5.2 コンポーネント別の並列化戦略

Wide-EP デプロイメントでは、モデルの各コンポーネントに最適な並列化を適用する:

| コンポーネント | 並列化方式 | 理由 |
|-------------|----------|------|
| **Attention (MLA)** | DP Attention | KV キャッシュの複製を排除。各ランクが独立にアテンションを実行 |
| **Dense FFN** | DP (TP より優先) | 通信オーバーヘッド 50% 削減。アライメント問題の回避 |
| **MoE FFN** | EP | Expert を全ランクに分散配置。all-to-all でトークンをルーティング |
| **LM Head** | DP | 語彙並列より実装がシンプルでメモリ効率的 |

### 5.3 メモリ効率

TP と比較した Wide-EP の最大の利点は **有効バッチサイズの拡大** である。TP ではアテンション層の潜在射影が全ランクに複製され、KV キャッシュメモリが圧迫される。Wide-EP の DP Attention では各ランクが独立した KV キャッシュを持ち、より多くのリクエストを同時に処理できる。

### 5.4 All-to-All 通信バックエンド

MoE 層ではトークンが expert のあるランクに送信され、処理後に元のランクに戻る。この all-to-all 通信には 3 つのバックエンドが利用可能:

1. **DeepEP**: DeepSeek が開発した高性能カーネル
   - **Normal Dispatch**: プリフィル向け。スループット最大化。動的形状をサポートするが CUDA Graph と非互換
   - **Low-Latency Dispatch**: デコード向け。レイテンシ最小化。固定メモリ事前割当が必要だが CUDA Graph をサポート
2. **Perplexity MoE カーネル**: Perplexity が開発した代替実装
3. **NCCL ベースの AllGather-ReduceScatter**: 汎用的だが専用カーネルより低効率

### 5.5 DeepGEMM との連携

Expert の GEMM 演算には **DeepGEMM** を使用する:

- **Contiguous Layout**: プリフィル向け。動的形状に対応。Normal Dispatch 後にカスタム Triton permutation カーネルで整列
- **Masked Layout**: デコード向け。固定形状 + マスクテンソル。Low-Latency Dispatch + CUDA Graph と組み合わせ

---

## 6. Dual Batch Overlap (DBO)

### 6.1 動機

Wide-EP デプロイメントでは、MoE 層の **all-to-all 通信**（Dispatch: トークン送信、Combine: 結果受信）が GPU の idle 時間を生む。プロファイリングでは、この通信が全ステップ時間の大きな割合を占めることが確認されている。

### 6.2 手法

DBO はバッチを 2 つの **マイクロバッチ** に分割し、一方の通信と他方の演算を **オーバーラップ** させる:

```
Time →
μBatch 1: [Attn] [MoE Dispatch₁] [          wait          ] [MoE Compute₁] [MoE Combine₁]
μBatch 2:                          [Attn] [MoE Dispatch₂]   [    wait     ] [MoE Compute₂] [MoE Combine₂]
                                           ↑ overlap ↑
```

より正確には:

1. **全ランクで cross-rank `all_reduce`** を実行し、各ランクの現在のトークン数を集約。マイクロバッチングが有効かどうかの閾値判定を行う（`--dbo-decode-token-threshold` で設定可能）
2. メインスレッドがマイクロバッチ用の **ワーカースレッド** を生成し、CUDA Graph キャプチャで各マイクロバッチを処理
3. MoE のモジュラーな基底クラスがマイクロバッチワーカーの **起動タイミングを調整**し、GPU 演算中に `yield` して次のワーカーに制御を渡す

### 6.3 実装の課題と UBatch 抽象

DBO の実装における重要な抽象化:

- **`UBatchWrapper`**: マイクロバッチの入力テンソルを管理するラッパー
- **`UBatchContext`**: マイクロバッチ実行のコンテキスト情報を保持

コードの重複を避けるため、単一マイクロバッチのロジックフローで処理を記述し、**yield ポイント** で制御を切り替える設計になっている。

### 6.4 性能改善

LMSYS の実測では、Two-Batch Overlap (TBO、DBO と同等の手法) により:

- **プリフィル: 27〜35% の高速化**
- デバイスあたり 16,384 トークンをサポート可能に（TBO なしでは 8,192 で OOM）→ **最大 40.5% のスループット向上**

---

## 7. Expert Parallel Load Balancing (EPLB)

### 7.1 問題

MoE モデルの expert は学習時にバランスが取られているが、推論時の実ワークロードではトークンの routing パターンが偏り、**expert 間の負荷不均衡** が生じる。あるランクに人気の expert が集中すると、そのランクがボトルネックとなり全体のスループットが低下する。

### 7.2 解決策: 動的再配置

vLLM は DeepSeek の **EPLB (Expert Parallel Load Balancing)** を実装している:

1. **統計収集**: 各 MoE forward pass でトークン単位の負荷を記録
2. **スライディングウィンドウ集約**: EP ランク全体の統計をウィンドウサイズ分蓄積
3. **再配置計算**: 指定インターバルで新しい logical-to-physical expert マッピングを計算
4. **重み再配置**: モデルを停止させずにシームレスに expert の重みをシャッフル

設定方法:

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
  --enable-eplb \
  --eplb-window-size 100 \
  --eplb-rebalance-interval 1000 \
  --eplb-num-redundant-experts 32
```

### 7.3 冗長 Expert

EPLB は **冗長 expert** の概念をサポートする。人気の expert を複数ランクに複製することで、非べき乗の EP サイズ（例: EP12, EP72）でも効率的な負荷分散を実現する。

DeepSeek-R1 の場合、256 base expert + 32 redundant expert = **288 expert** で EP72 構成を取ることができる。

### 7.4 性能影響

LMSYS の実測結果:

- **プリフィル: 1.49× 高速化**
- **デコード: 2.54× 高速化**

Expert の不均衡が解消されることで、全ランクの GPU 利用率が均一化される。

---

## 8. Prefill/Decode Disaggregation (P/D)

### 8.1 動機

プリフィルとデコードは本質的に異なる性能特性を持つ:

| フェーズ | 特性 | ボトルネック |
|---------|------|------------|
| **Prefill** | 大量トークンの並列処理 | 演算律速 (compute-bound) |
| **Decode** | 1 トークンずつ逐次生成 | メモリ帯域律速 (memory-bound) |

これらを同一インスタンスで処理すると、compute-bound なプリフィルが memory-bound なデコードの ITL を悪化させる（**prefill interference**）。

Expert Parallel デプロイメントではこの問題がさらに深刻化する。MoE 層では全ランクが同期して all-to-all 通信を行うため、**1 つのランクがプリフィルで重い計算をしていると、EP グループ全体の forward pass が遅延する**。

### 8.2 アーキテクチャ

P/D Disaggregation は推論インスタンスをプリフィル専用とデコード専用に分離する:

```
          ┌─────────────────┐
          │   Router/LB     │
          └──────┬──────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Prefill      │  │ Decode       │
│ Instances    │  │ Instances    │
│ (EP32, 4node)│  │ (EP72, 9node)│
└──────┬───────┘  └──────▲───────┘
       │   KV Cache      │
       └─── Transfer ────┘
          (NIXL/RDMA)
```

- **プリフィルインスタンス**: DeepEP Normal Dispatch（高スループット）を使用
- **デコードインスタンス**: DeepEP Low-Latency Dispatch + CUDA Graph を使用
- **KV キャッシュ転送**: NVIDIA **NIXL** ライブラリによる non-blocking RDMA 転送

### 8.3 Connector インターフェース

vLLM の P/D は **Connector** 抽象を通じて KV キャッシュを交換する:

1. **スケジューラ側**: `connector.get_num_new_matched_tokens` で外部 KV キャッシュの有無を確認
   - プリフィルインスタンス: 常に 0（自分で計算する）
   - デコードインスタンス: キャッシュヒットがあれば再計算不要
2. **メモリ確保後**: `connector.update_state_after_alloc` でリクエストを登録
3. **メタデータ構築**: `connector.build_connector_meta`
   - プリフィル: `is_store=True`（KV を保存）
   - デコード: `is_store=False`（KV をロード）
4. **forward pass のコンテキストマネージャ**:
   - デコード: `kv_connector.start_load_kv` で KV をロード（初回ステップのみ）
   - プリフィル: `kv_connector.wait_for_save` で KV をアップロード

本番環境では **LMCache** + NVIDIA NIXL がバックエンドとして使用される。開発・デバッグ用にはローカルファイルシステムベースの `SharedStorageConnector` も用意されている。

### 8.4 独立スケーリング

P/D の大きな利点は、プリフィルとデコードのインスタンス数を**独立にオートスケール**できることである。プリフィル負荷が高い（長いプロンプト）場合はプリフィルインスタンスを、デコード負荷が高い（長い生成）場合はデコードインスタンスを増やせる。

---

## 9. マルチ GPU アーキテクチャ

### 9.1 UniProcExecutor vs MultiProcExecutor

- **UniProcExecutor**: 単一 GPU。Worker 1 つでシンプルに実行
- **MultiProcExecutor**: 複数 GPU。TP/PP/EP をサポート

MultiProcExecutor の仕組み:

1. 各ランクにデーモンプロセスを `WorkerProc.make_worker_process` で起動
2. 各ワーカーは `rpc_broadcast_mq`（親→子）と `worker_response_mq`（子→親）の 2 つの共有メモリキューで通信
3. ランタイム: ワーカーは `rpc_broadcast_mq.dequeue` でビジーループし、タスク実行後に結果を `worker_response_mq` に返送
4. 親プロセスは全ワーカーにノンブロッキングで配信し、出力ランクのレスポンスキューで待機

### 9.2 分散サービングアーキテクチャ

本番環境では **DPEngineCoreProc** がノード間でデータ並列にエンジンコアを起動する:

- 各エンジンコアは **メイン・入力・出力** の 3 スレッド構成
- **入力スレッド**: ソケットからリクエストをデコード
- **メインスレッド**: エンジンステップを実行
- **出力スレッド**: 結果をクライアントに返送

ロードバランシングのスコアリング:

$$\text{score} = |\text{waiting}| \times 4 + |\text{running}|$$

待機中のリクエストに高い重みを付けることで、backlog の少ないエンジンに優先的にルーティングする。

---

## 10. カーネルレベルの最適化

vLLM の 2.2k tok/s/GPU 達成に貢献した具体的なカーネル最適化:

| 最適化 | 効果 |
|--------|------|
| **SiLU-Mul-Quant Fusion** | 3 カーネルを 1 つに融合。メモリ帯域の節約 |
| **Cutlass QKV カーネル** | アテンションの Q/K/V 射影を最適化された GEMM で実行 |
| **TP Attention バグ修正** | 正しい並列化による性能回復 |
| **DeepGEMM デフォルト有効化** | MoE expert の GEMM 最適化 |
| **DeepSeek-R1 専用 SiLU カーネル** | モデル固有の活性化関数最適化 |

---

## 11. デプロイメントフレームワーク

### 11.1 llm-d

**Kubernetes ネイティブ** の分散推論スタックで、Wide-EP の構成を再現するためのドキュメントが充実している。vLLM + DeepSeek の本番デプロイに適している。

### 11.2 NVIDIA Dynamo

NVIDIA 製フレームワークで、以下の機能を提供:

- **KV-aware routing**: KV キャッシュの局所性を考慮したリクエストルーティング
- **KV Block Manager**: キャッシュのオフロードと再利用
- **Dynamic Load Matching Planner**: SLO を満たしつつスループットを最大化する動的負荷分散
- vLLM と Wide-EP をネイティブサポート

### 11.3 Ray Serve LLM

**Anyscale** の Ray Serve 上で vLLM を動かすパターン:

- Prefill/Decode Disaggregation の first-class サポート
- DP Attention と Prefix Cache Affinity Routing
- NIXL と LMCache コネクタによる KV 転送
- **フェーズ独立のオートスケーリング**

---

## 12. 性能ベンチマーク

### 12.1 vLLM (H200, Coreweave)

| 構成 | スループット |
|------|-----------|
| Wide-EP (最適化前) | ~1.5k tok/s/GPU |
| Wide-EP + DBO + EPLB + カーネル最適化 | **2.2k tok/s/GPU** |

### 12.2 LMSYS (96 H100, Atlas Cloud)

| フェーズ | 構成 | スループット (tok/s/node) |
|---------|------|------------------------|
| Prefill (1K input) | EP32, 4 nodes | 57,674 |
| Prefill (2K input) | EP32, 4 nodes | 54,543 |
| Prefill (4K input) | EP32, 4 nodes | 50,302 |
| Decode (2K input) | EP72, 9 nodes | 22,282 |

TP16 ベースラインと比較して、プリフィルで **3.3×**、デコードで **5.2×** の高速化。

### 12.3 GB200 (Blackwell)

最新のベンチマークでは、GB200 上で:

- **プリフィル: 26.2K TPGS** (tokens per GPU second)
- **デコード: 10.1K TPGS**

（ワークロード: 2K input + 2K output）

---

## 13. 今後のロードマップ

vLLM の今後の開発方針:

- **Elastic Expert Parallelism**: ワークロードに応じて動的に EP サイズを変更
- **Long Context Serving**: 長系列入力の効率的な処理
- **CPU ベースの KV キャッシュ転送**: GPU メモリを節約しつつ P/D を実現
- **Full Determinism & Batch Invariance**: 再現性の保証
- **Large MoE Fusion**: MoE 演算のさらなるカーネル融合
- **FlashInfer 統合改善**: アテンションバックエンドの最適化
- **GB200 最適化**: Blackwell アーキテクチャへの本格対応
- **P/D における独立 TP サイズ**: プリフィルとデコードで異なる TP 度数の設定

---

## まとめ

vLLM の Large Scale Serving は、単一の銀の弾丸ではなく、**複数の最適化技術の組み合わせ** によって実現されている:

1. **PagedAttention + Continuous Batching** が基盤として高いメモリ効率とスループットを提供
2. **Wide-EP** が DeepSeek のような大規模 MoE モデルに対して、TP の限界を超えたスケーラビリティを実現
3. **Dual Batch Overlap** が all-to-all 通信のオーバーヘッドを隠蔽し、GPU 利用率を最大化
4. **EPLB** が推論時の expert 負荷不均衡を動的に解消
5. **P/D Disaggregation** がプリフィルとデコードの干渉を排除し、SLO 遵守とスループットを両立
6. **カーネルレベルの融合・最適化** が各レイヤーでの演算効率を底上げ

これらは独立に機能するだけでなく、**相互に強化し合う**。例えば、P/D Disaggregation によりプリフィルとデコードに異なる DeepEP Dispatch モードを適用でき、DBO と EPLB がデコードフェーズの通信・負荷分散を改善する。

推論最適化の研究においては、このようなシステム全体を俯瞰する視点と、個々のカーネル・通信パターンへの深い理解の**両方**が求められる。

---

## References

- [vLLM Blog: Large Scale Serving](https://blog.vllm.ai/2025/12/17/large-scale-serving.html)
- [Aleksa Gordic: vLLM Internals](https://www.aleksagordic.com/blog/vllm)
- [LMSYS: Deploying DeepSeek with PD Disaggregation on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [vLLM Documentation: Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM Documentation: Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/dbo/)
- [Red Hat: Scaling DeepSeek-style MoEs with vLLM and llm-d](https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep)
- [vLLM Blog: DeepSeek-R1 on GB200 (Part I)](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)
