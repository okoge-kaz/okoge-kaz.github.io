---
title: "NVFP4 — 4-bit 推論の精度と効率を両立する NVIDIA の新量子化フォーマット"
date: 2026-02-08
tags: ["LLM", "Inference", "Quantization", "NVFP4", "NVIDIA", "Blackwell", "TensorRT-LLM"]
description: "NVIDIA が Blackwell アーキテクチャ向けに設計した 4-bit 浮動小数点フォーマット NVFP4 の技術詳細を解説する。データフォーマットの仕様、Two-Level Scaling、Quantization-Aware Distillation (QAD) による精度回復、そしてプリトレーニングへの適用まで、包括的にまとめる。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

LLM の推論コストは、モデルの大規模化に伴い深刻な課題となっている。数百億〜数兆パラメータのモデルを実用的なレイテンシで提供するには、**メモリ帯域幅**と**計算効率**の両面から最適化が不可欠である。

量子化（Quantization）は、モデルの重みやアクティベーションの数値精度を下げることで、メモリ使用量と計算コストを削減する手法である。FP32 → FP16 → BF16 → FP8 と精度の低下が進んできたが、さらに **4-bit** まで下げると精度劣化が深刻になるという問題があった。

NVIDIA が提案する **NVFP4** は、Blackwell アーキテクチャのネイティブサポートと独自の Two-Level Scaling 機構を組み合わせることで、4-bit の効率性と BF16 に近い精度の両立を目指すフォーマットである。

本記事では以下を解説する:

1. LLM 推論における量子化の背景
2. NVFP4 フォーマットの技術仕様
3. Two-Level Scaling の仕組み
4. MXFP4 との比較
5. Quantization-Aware Distillation (QAD) による精度回復
6. NVFP4 プリトレーニングへの応用
7. ハードウェア・ソフトウェアエコシステム

参考文献:
- [Nemotron QAD](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
- [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery (arXiv:2601.20088)](https://arxiv.org/abs/2601.20088)
- [NVFP4 Trains with Precision of 16-bit and Speed and Efficiency of 4-bit](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

## LLM 推論と量子化の背景

### なぜ量子化が必要か

LLM 推論の Decode フェーズは **Memory bandwidth-bound** である。各ステップで1トークンを生成するために、モデルの全重みを GPU メモリから読み出す必要があり、演算量に対するメモリアクセスの比率（Arithmetic Intensity）が極めて低い。

重みの精度を半分にすれば、メモリからの読み出しデータ量も半分になり、理論上のスループットは2倍になる。これが量子化の基本的な動機である。

| 精度 | ビット幅 | 対 FP16 メモリ削減 | 主な用途 |
|---|---|---|---|
| FP32 | 32 bit | 0.5x（増加） | 学習（旧世代） |
| BF16 / FP16 | 16 bit | 1x（基準） | 学習・推論 |
| FP8 (E4M3/E5M2) | 8 bit | 2x | 学習・推論 (Hopper+) |
| INT8 | 8 bit | 2x | 推論（Weight-only） |
| INT4 / FP4 | 4 bit | 4x | 推論 |

### Weight-Only vs. Weight-Activation 量子化

4-bit 量子化には大きく2つのアプローチがある:

- **Weight-Only 量子化**: 重みのみを 4-bit に量子化し、アクティベーションは FP16/BF16 のまま保持する。GPTQ、AWQ などが代表的。メモリ削減は得られるが、演算自体は FP16 で行われるため **Tensor Core の 4-bit 演算性能を活用できない**。
- **Weight-Activation 量子化**: 重みとアクティベーションの両方を 4-bit に量子化する。Tensor Core の FP4 演算を直接利用でき、**BF16 比で 4 倍の FLOPS** を達成可能。ただし、アクティベーションの量子化は精度劣化が大きく技術的に困難。

NVFP4 は **Weight-Activation 量子化** を採用しており、Blackwell Tensor Core の FP4 演算能力をフルに活用する設計である。

## NVFP4 フォーマットの技術仕様

### ビットレイアウト: E2M1

NVFP4 は **E2M1**（2-bit Exponent, 1-bit Mantissa）の 4-bit 浮動小数点フォーマットである:

```
┌──────┬──────────┬──────────┐
│ Sign │ Exponent │ Mantissa │
│ 1bit │  2 bits  │  1 bit   │
└──────┴──────────┴──────────┘
```

この 4-bit で表現可能な値は以下の通り:

| ビット列 | 値 |
|---|---|
| 0 000 | +0.0 |
| 0 001 | +0.5 |
| 0 010 | +1.0 |
| 0 011 | +1.5 |
| 0 100 | +2.0 |
| 0 101 | +3.0 |
| 0 110 | +4.0 |
| 0 111 | +6.0 |

符号ビットにより負の値も同様に表現でき、合計 **16 種類の離散値**（$\pm$ {0, 0.5, 1, 1.5, 2, 3, 4, 6}）をカバーする。値の範囲は **-6.0 〜 +6.0** である。

> 4-bit で表現できる値がたった 16 種類しかないため、そのままではモデルの重みやアクティベーションの分布を正確に捉えることは不可能である。この問題を解決するのが次節で説明する **Two-Level Scaling** である。

### 実効ビット幅

NVFP4 の実効的な記憶コストは、純粋な 4-bit ではなく、スケーリングファクタの分も含めて **約 4.5 bits/value** となる:

$$\text{effective bits} = 4 + \frac{8}{16} = 4.5 \text{ bits/value}$$

16 要素ごとに 8-bit（E4M3）のスケーリングファクタが1つ付与されるためである。それでも FP8（8 bits）と比較して **約 1.8 倍** のメモリ削減を実現する。

## Two-Level Scaling

NVFP4 の精度を支える核心技術が **Two-Level Scaling**（2段階スケーリング）である。

### Level 1: Micro-Block Scaling

テンソルの値を **16 要素ごとのブロック（Micro-Block）** に分割し、各ブロックに **E4M3 (FP8)** 精度のスケーリングファクタを割り当てる。

```
テンソル: [v0, v1, ..., v15 | v16, v17, ..., v31 | ...]
           ─── block 0 ───   ──── block 1 ────
           scale: s0 (E4M3)   scale: s1 (E4M3)
```

復元（Dequantization）は以下の式で行われる:

$$x_{\text{dequant}} = x_q \times s_{\text{block}}$$

ここで $x_q$ は 4-bit 量子化値（E2M1）、$s_{\text{block}}$ は E4M3 のブロックスケーリングファクタである。

### Level 2: Tensor-Level Scaling

テンソル全体に対して **FP32 (E8M23)** のスケーリングファクタを1つ適用する。この2段目のスケールにより、ブロックレベルのスケーリングファクタ自体のダイナミックレンジが E4M3 の範囲内に収まるように正規化される。

```
テンソル全体
├── tensor_scale: S (FP32)
│
├── block 0: [q0, q1, ..., q15] + block_scale: s0 (E4M3)
├── block 1: [q16, q17, ..., q31] + block_scale: s1 (E4M3)
├── ...
└── block N: [...] + block_scale: sN (E4M3)

復元: x = q × s_block × S
```

### なぜ 2 段階が必要か

E4M3 のダイナミックレンジは $\pm 448$ であり、これだけではテンソル全体の値域をカバーするには不十分な場合がある。テンソルレベルの FP32 スケールで事前に正規化することで、各ブロックスケールが E4M3 の範囲内で精度を最大限に発揮できるようになる。

この階層的なスケーリングにより、16 要素という細粒度でデータの局所的な分布に適応しつつ、テンソル全体のスケールも正確に保持する。

## MXFP4 との比較

NVFP4 は OCP（Open Compute Project）標準の **MXFP4** と比較して、以下の2つの重要な差異を持つ:

### 1. ブロックサイズ: 32 → 16

| | MXFP4 | NVFP4 |
|---|---|---|
| ブロックサイズ | 32 要素 | **16 要素** |

ブロックサイズが半分であるため、**局所的な値の分布変動に対してより精密に適応**できる。テンソル内に外れ値が存在する場合、大きなブロックではその外れ値のためにブロック全体のスケールが引き上げられ、他の値の精度が犠牲になる。ブロックサイズを 16 に縮小することで、外れ値の影響範囲を限定し、量子化誤差を削減する。

### 2. スケーリングファクタ: E8M0 → E4M3

| | MXFP4 | NVFP4 |
|---|---|---|
| スケール精度 | E8M0（2のべき乗のみ） | **E4M3（小数精度あり）** |
| スケール範囲 | $2^{-127}$ 〜 $2^{127}$ | $\pm 448$（FP32 テンソルスケールで補完） |

MXFP4 の E8M0 スケーリングファクタは **2のべき乗** に制限される（1, 2, 4, 8, 16, ...）。これは実装がシンプルだが、ブロック内の値の最適なスケールが 2 のべき乗でない場合に量子化誤差が増大する。

NVFP4 の E4M3 スケーリングファクタは **小数値を表現可能**であり、ブロック内のデータ分布に対してより最適なスケールを選択できる。

この差異は量子化誤差（MSE）に顕著に現れる:

| スケールフォーマット | 量子化 MSE |
|---|---|
| E8M0（MXFP4） | ~0.72 |
| E4M3（NVFP4） | **~0.08** |

NVFP4 は MXFP4 と比較して **約 9 倍** 量子化誤差が小さい。

## Post-Training Quantization (PTQ)

### 推論パイプラインへの適用

NVFP4 を推論で使用する最もシンプルな方法は **Post-Training Quantization (PTQ)** である。学習済みモデル（BF16 / FP8）を対象として、重みとアクティベーションの量子化パラメータをキャリブレーションデータを用いて決定する。

PTQ のワークフロー:

1. BF16 / FP8 の学習済みモデルを用意
2. キャリブレーションデータで各レイヤーのアクティベーション分布を収集
3. ブロックスケーリングファクタとテンソルスケーリングファクタを計算
4. 重みを NVFP4 フォーマットに変換
5. 推論時にアクティベーションもオンラインで NVFP4 に量子化

### PTQ の限界

PTQ は追加学習が不要でシンプルだが、**小規模モデルでは無視できない精度劣化**が生じる。特に:

- パラメータ数が少ないモデルでは、個々の重みの量子化誤差がモデル全体の出力に大きく影響する
- アクティベーションの外れ値が多いレイヤーでは、16 要素ブロック内の値の分散が大きくなり精度が低下する
- 大規模モデル（例: DeepSeek-R1-0528）では PTQ でも **1% 以下の精度劣化** に抑えられるケースが多い

## Quantization-Aware Distillation (QAD)

PTQ の精度劣化を回復するために NVIDIA が提案するのが **Quantization-Aware Distillation (QAD)** である（[arXiv:2601.20088](https://arxiv.org/abs/2601.20088)）。

### 概要

QAD は知識蒸留（Knowledge Distillation）を量子化モデルの精度回復に適用する手法である。BF16 の高精度モデル（Teacher）から NVFP4 量子化モデル（Student）へ知識を転送する。

```
┌──────────────────────────┐
│   Teacher Model (BF16)   │  ← Frozen（パラメータ固定）
│   高精度・大メモリ         │
└──────────┬───────────────┘
           │ Soft Targets (logits)
           ▼
     ┌─────────────┐
     │  KL Divergence│
     │     Loss      │
     └─────┬───────┘
           │ Gradient
           ▼
┌──────────────────────────┐
│  Student Model (NVFP4)   │  ← 学習対象
│  低精度・省メモリ          │
└──────────────────────────┘
```

### 学習目的関数

QAD の損失関数は、Teacher と Student の出力分布間の **KL ダイバージェンス** を最小化する:

$$\mathcal{L}_{\text{QAD}} = D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})$$

ここで $P_{\text{teacher}}$ と $P_{\text{student}}$ はそれぞれ Teacher と Student の出力 logits から計算されるトークン確率分布である。

従来の量子化対応学習（QAT）がハードラベル（正解トークン）に対する Cross-Entropy Loss を使用するのに対し、QAD は **ソフトターゲット**（Teacher の出力確率分布）を用いる。ソフトターゲットには、正解トークン以外のトークンに対する確率情報（"dark knowledge"）が含まれており、これにより Student は Teacher の振る舞いをより正確に模倣できる。

### QAD vs. QAT

| | QAT | QAD |
|---|---|---|
| 学習信号 | ハードラベル（正解） | ソフトターゲット（Teacher logits） |
| Teacher モデル | 不要 | 必要（BF16 モデル） |
| Post-Training との相性 | 低い（SFT/RL/Merging との組合せが複雑） | **高い**（単一ステージ） |
| 学習安定性 | RL 後のモデルで不安定 | **安定** |
| データ要件 | 元の学習データが必要 | **部分的なデータでも可** |

### QAT の課題

現代の LLM は単純なプリトレーニングだけでなく、**多段階のポストトレーニングパイプライン**を経ている:

1. **SFT**（Supervised Fine-Tuning）
2. **RL**（Reinforcement Learning from Human Feedback 等）
3. **Model Merging**（複数チェックポイントの統合）

QAT をこのパイプラインに組み込むには、各ステージで量子化を意識した学習を行う必要があり、エンジニアリングの複雑さが増大する。特に RL ステージでは量子化による勾配ノイズが学習の不安定性を引き起こしやすい。

QAD はこれらの **ポストトレーニング完了後のモデル** に対して、**単一の蒸留ステージ**として適用できる。パイプラインの各ステージに手を加える必要がなく、既存のワークフローへの組み込みが容易である。

### 評価結果

QAD は以下のモデルで BF16 に近い精度への回復を実証している:

- **AceReason Nemotron** — 推論特化モデル
- **Nemotron 3 Nano** (30B-A3B) — Hybrid MoE モデル
- **Nemotron Nano V2** — 第2世代
- **Nemotron Nano V2 VL** — Vision-Language モデル
- **Llama Nemotron Super v1** — Llama ベースモデル

特筆すべきは、**Vision-Language モデル（VLM）にも適用可能**である点であり、テキストのみの LLM に限定されない汎用性を持つ。

## NVFP4 プリトレーニング

NVFP4 は推論だけでなく、**プリトレーニング**にも適用できる。NVIDIA は 12B Hybrid Mamba-Transformer モデルを 10 兆トークンで学習し、NVFP4 が FP8 と同等の精度を達成することを示している。

### プリトレーニングにおける課題

学習時に FP4 を使用するには、推論時よりもはるかに高い精度の維持が求められる。Forward pass だけでなく **Backward pass（勾配計算）** でも量子化が行われるため、勾配の劣化が学習の安定性と最終精度に直結する。

### NVFP4 プリトレーニングの5つの要素技術

#### 1. Micro-Block Scaling

推論と同様に 16 要素ブロック + E4M3 スケーリングを使用する。学習時にはアクティベーションだけでなく**勾配**に対しても Micro-Block Scaling を適用する。

#### 2. High-Precision Block Encoding

E4M3 スケーリングファクタにより、E8M0（2のべき乗のみ）と比較してブロック内の値をより精密に表現する。学習中の微細な勾配の変化を捉えるために重要。

#### 3. Random Hadamard Transform

アクティベーションと勾配に **Random Hadamard 変換**を適用し、分布をよりガウス分布に近い形に変形する。

LLM のアクティベーションには極端な外れ値が存在することが知られている。外れ値があるとブロック内のダイナミックレンジが過大になり、他の値の量子化精度が犠牲になる。Hadamard 変換はこれらの外れ値を分散させ、量子化に適した滑らかな分布を生成する。

#### 4. 2D Block Quantization

Forward pass と Backward pass で**一貫した量子化**を保証するために、2次元ブロックベースの量子化を採用する。行列の行方向と列方向の両方でブロックを構成し、Forward/Backward 間の量子化の不整合（signal distortion）を低減する。

#### 5. Stochastic Rounding

確率的丸め処理（Stochastic Rounding）を使用する。決定的な丸め（Round-to-Nearest）では、常に同じ方向に丸められるバイアスが生じ、勾配の蓄積に系統的な誤差が発生する。

Stochastic Rounding では、値が隣接する2つの表現可能な値の間にあるとき、その距離に比例した確率で上または下に丸める:

$$P(\text{round up}) = \frac{x - \lfloor x \rfloor_q}{\lceil x \rceil_q - \lfloor x \rfloor_q}$$

これにより丸め誤差の**期待値がゼロ**になり、勾配の流れが長期的に保全される。

### 学習結果

12B Hybrid Mamba-Transformer モデルの 10 兆トークン学習において:

- **Validation Loss**: NVFP4 は FP8 ベースラインとほぼ一致する損失曲線を示し、学習の全フェーズを通じて安定して収束
- **下流タスク精度**: MMLU Pro、コード生成、数学、常識推論、多言語タスクの全領域で FP8 と同等の精度を達成
- **GB300 での GEMM 速度**: Hopper 世代比で **7 倍** の GEMM 高速化

## ハードウェアとソフトウェアの対応状況

### Blackwell アーキテクチャ

NVFP4 は NVIDIA **Blackwell** 世代（B200, GB200, GB300）の第5世代 Tensor Core でネイティブにサポートされる。ハードウェアが Micro-Block FP4 データの**グルーピング、動的スケーリング、4-bit 行列演算**を自動的に処理する。

性能指標:

| 指標 | 値 |
|---|---|
| 対 BF16 FLOPS 向上 | **4x** |
| 対 FP8 メモリ削減 | **1.8x** |
| 対 FP16 メモリ削減 | **3.5x** |
| 対 Hopper エネルギー効率 | **25-50x** |
| GEMM 速度（対 Hopper） | **7x** |

> **コンシューマ GPU**: RTX 5090 も Blackwell アーキテクチャを採用しており、NVFP4 をサポートする。大規模モデルのローカルデプロイが現実的になる。

### 推論フレームワーク

| フレームワーク | NVFP4 対応状況 |
|---|---|
| TensorRT-LLM | Early support |
| vLLM | Early support |
| SGLang | 対応予定 |
| TensorRT Model Optimizer | PTQ / QAT 対応 |
| LLM Compressor | 量子化ワークフロー対応 |

### 事前量子化モデル

HuggingFace 上で以下の NVFP4 量子化済みチェックポイントが公開されている:

- DeepSeek-R1-0528
- Llama 3.1-405B
- FLUX.1-dev

## まとめ

NVFP4 は単なる「4-bit に精度を落とす」フォーマットではなく、以下の技術スタックが一体となって精度と効率を両立させている:

1. **E2M1 フォーマット + Two-Level Scaling** — 4-bit の制約下で最大限の表現力を確保
2. **16 要素 Micro-Block + E4M3 スケール** — MXFP4 比で約 9 倍の量子化誤差削減
3. **Quantization-Aware Distillation (QAD)** — ポストトレーニング後のモデルに対して単一ステージで BF16 精度に回復
4. **Random Hadamard Transform + Stochastic Rounding** — プリトレーニングでの FP8 相当精度を実現
5. **Blackwell Tensor Core のネイティブサポート** — BF16 比 4 倍の FLOPS、Hopper 比 7 倍の GEMM 高速化

LLM 推論の量子化は FP8 が主流となりつつあるが、NVFP4 の登場により **4-bit 推論が実用的な選択肢** となった。Blackwell 世代の GPU が普及するにつれ、NVFP4 は推論コスト削減の標準的な手法になる可能性がある。
