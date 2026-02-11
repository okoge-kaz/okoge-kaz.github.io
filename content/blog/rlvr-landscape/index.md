---
title: "RLVR の全体像 — アルゴリズム・安定化・フレームワーク設計"
date: 2026-02-08
tags: ["LLM", "RLVR", "GRPO", "GSPO", "PPO", "Post-Training", "Reinforcement Learning", "MoE"]
description: "Reinforcement Learning with Verifiable Rewards (RLVR) の全体像を整理する。Post-Training パイプラインの基礎から、GRPO/GSPO のアルゴリズム比較、Off-Policy 学習の安定化手法、そして VERL・OpenRLHF・AReaL 等の分散 RL フレームワークの設計思想までを解説する。"
draft: true
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## はじめに

DeepSeek-R1 以降、**Reinforcement Learning with Verifiable Rewards (RLVR)** は LLM の推論能力を引き出す中核手法として急速に注目を集めている。RLVR は、人手による嗜好ラベルの代わりに **検証可能な報酬**（数学問題の正誤、コードのテスト通過など）を用いることで、スケーラブルかつ客観的な RL 学習を実現する。

本記事では、RLVR に関する複数のリソースを横断的に整理し、以下の4つの軸で全体像を描く:

1. **Post-Training パイプラインの基礎** — SFT から RL へ
2. **アルゴリズム** — GRPO・GSPO の設計と比較
3. **Off-Policy 学習の安定化** — トークンレベル目的関数の理論的正当化
4. **分散 RL フレームワークの設計** — VERL・OpenRLHF・AReaL 等の比較

### 参考リソース

| リソース | 概要 |
|---|---|
| [Post-Training 101 (Tokens for Thoughts)](https://tokens-for-thoughts.notion.site/post-training-101) | Han Fang & Karthik Abinav Sankararaman (Meta) による Post-Training 入門 |
| [Off-Policy RL (Feng Yao)](https://fengyao.notion.site/off-policy-rl) | 効率的 RL フレームワークにおける Off-Policy 問題の分析 |
| [Anatomy of RL Frameworks (Hanif Leo)](https://www.hanifleo.com/anatomy-of-rl-frameworks/) | VERL / OpenRLHF / AReaL 等の RL フレームワーク比較 |
| [GSPO (Zheng et al., 2025)](https://arxiv.org/abs/2507.18071) | Group Sequence Policy Optimization — 系列レベル最適化 |
| [Stabilizing RL for LLMs (Zheng et al., 2025)](https://arxiv.org/abs/2512.01374) | トークンレベル目的関数の理論的分析と MoE 安定化手法 |

---

## Post-Training パイプラインの基礎

LLM の Post-Training は大きく2つのステージから構成される。

### Supervised Fine-Tuning (SFT)

事前学習済みモデルに対して、高品質な instruction-response ペアを用いて教師あり学習を行う。SFT はモデルに **フォーマットの学習**（指示に対して適切な形式で応答する能力）と**基礎的な知識の整理**を与える。

ただし SFT には本質的な限界がある。SFT は正解データの **模倣** に過ぎず、モデルが「なぜその回答が良いのか」を理解する仕組みは持たない。

### Reinforcement Learning (RL)

RL は SFT の限界を超え、モデルが **報酬シグナルに基づいて自律的に戦略を最適化する** ことを可能にする。

**RLHF (RL from Human Feedback)** では、人手の嗜好ラベルから学習した Reward Model を報酬として用いる。一方 **RLVR (RL with Verifiable Rewards)** では、数学の正誤判定やコードのテスト実行結果など、**客観的に検証可能な報酬** を直接使用する。

$$
J(\theta) = \mathbb{E}\_{y \sim \pi\_\theta(\cdot|x)} \left[ R(x, y) \right]
$$

RLVR の利点は明確である:

- **Reward Model が不要**: 報酬は決定的に計算されるため、Reward Model の学習コストとバイアスを排除できる
- **スケーラブル**: 数学・コード等の自動検証可能なドメインでは、報酬ラベルを大量かつ安価に生成できる
- **報酬ハッキングの抑制**: Reward Model の脆弱性を突くような行動が生じにくい

---

## アルゴリズム: GRPO と GSPO

### GRPO (Group Relative Policy Optimization)

DeepSeek が提案した GRPO は、PPO から Value Function（Critic）を除去し、**同一プロンプトに対する複数の生成結果のグループ内比較** で Advantage を推定する。

$$
\hat{A}\_i = \frac{r(x, y\_i) - \text{mean}(\{r(x, y\_j)\}\_{j=1}^G)}{\text{std}(\{r(x, y\_j)\}\_{j=1}^G)}
$$

GRPO の目的関数は **トークンレベル** で定義される:

$$
J\_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum\_{i=1}^G \frac{1}{|y\_i|} \sum\_{t=1}^{|y\_i|} \min\left( r\_t^i(\theta) \hat{A}\_i, \; \text{clip}(r\_t^i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}\_i \right) \right]
$$

ここで $r\_t^i(\theta) = \frac{\pi\_\theta(y\_{i,t} | x, y\_{i,<t})}{\pi\_{\theta\_{\text{old}}}(y\_{i,t} | x, y\_{i,<t})}$ はトークンレベルの importance ratio である。

**GRPO の問題点**: トークンレベルの importance ratio は各トークン位置で独立に変動するため、系列長が長くなるにつれて **勾配の分散が蓄積** し、学習が不安定になる。特に MoE モデルでは、expert routing がトークンごとに変化するため、この問題が深刻化する。

### GSPO (Group Sequence Policy Optimization)

Qwen チームが提案した GSPO は、GRPO の不安定性を根本的に解決するため、**系列レベル** で importance ratio とクリッピングを定義する。

$$
s\_i(\theta) = \left( \frac{\pi\_\theta(y\_i | x)}{\pi\_{\theta\_{\text{old}}}(y\_i | x)} \right)^{1/|y\_i|}
$$

長さ正規化 $(\cdot)^{1/|y_i|}$ により、異なる系列長間で importance ratio の数値範囲が統一される。

$$
J\_{\text{GSPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum\_{i=1}^G \min\left( s\_i(\theta) \hat{A}\_i, \; \text{clip}(s\_i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}\_i \right) \right]
$$

### GRPO vs GSPO: 核心的な違い

| 観点 | GRPO | GSPO |
|---|---|---|
| Importance ratio | トークンレベル $r_t^i(\theta)$ | 系列レベル $s_i(\theta)$ |
| クリッピング | 各トークン位置で独立に適用 | 系列全体に対して1回適用 |
| 勾配の分散 | トークン数に比例して蓄積 | 系列内で均一に分配 |
| MoE 安定性 | Routing Replay が必要 | 本質的にロバスト |
| 推論エンジンとの互換性 | トークン確率の同期が必要 | 系列尤度を直接利用可能 |

GSPO が系列レベルでクリッピングする結果、トークン単位ではGRPO の **2桁多い** トークンがクリップされる。にもかかわらず GSPO の方が高い学習効率を達成しており、GRPO のトークンレベル勾配がいかにノイジーであるかを示している。

### MoE モデルにおける安定性

MoE モデルでは、各トークンが異なる expert にルーティングされる。GRPO のトークンレベル importance ratio は expert routing の変化に直接影響を受けるため、**学習と推論で異なる expert が活性化** されると importance ratio が大きく乖離する。

GSPO は系列尤度のみに依存するため、個別トークンの expert routing 変化に影響されない。この性質により、GRPO で必要とされた **Routing Replay** を不要にし、メモリ使用量の削減とアルゴリズムの単純化を同時に達成する。

---

## Off-Policy 学習の安定化

効率的な RL 学習では、ロールアウト（生成）と学習を並列化・非同期化するため、学習時のポリシーがロールアウト時のポリシーと乖離する **Off-Policy** 問題が不可避的に発生する。

### トークンレベル目的関数の理論的正当化

Zheng et al. (2025) は、トークンレベルの代理目的関数 (surrogate objective) が真の系列レベル目的関数の **一次近似** として有効であることを示した。その有効性は2つの条件に依存する:

$$
\underbrace{\frac{\pi\_{\theta\_{\text{old}}}(y\_t | x, y\_{<t})}{\mu\_{\theta\_{\text{old}}}(y\_t | x, y\_{<t})}}_{\text{Training-Inference Discrepancy}} \times \underbrace{\frac{\pi\_\theta(y\_t | x, y\_{<t})}{\pi\_{\theta\_{\text{old}}}(y\_t | x, y\_{<t})}}_{\text{Policy Staleness}}
$$

1. **Training-Inference Discrepancy**: 学習エンジンと推論エンジンの数値的差異（FP8 vs BF16 の精度差、異なる計算カーネルなど）
2. **Policy Staleness**: ロールアウト時のポリシー $\pi_{\theta_{\text{old}}}$ と最適化対象のポリシー $\pi_\theta$ の乖離

この2つの要因が小さい場合にのみ、トークンレベルの代理目的関数は真の目的関数を適切に近似する。

### MiniRL: 最小限のベースライン

上記の理論に基づき、Zheng et al. は **MiniRL** を提案する。REINFORCE に2つの修正を加えた最小限のアルゴリズムである:

1. **Group Normalization** による Advantage 推定: $\hat{A}(x,y) = R(x,y) - \mathbb{E}\_{y' \sim \mu\_{\theta\_{\text{old}}}}[R(x,y')]$
2. **PPO スタイルのクリッピング**: importance ratio が閾値を超えた場合に勾配マスクを適用

$$
M\_t = \begin{cases} 0 & \text{if } \hat{A} > 0 \text{ and } r\_t > 1+\varepsilon\_{\text{high}} \\\\ 0 & \text{if } \hat{A} < 0 \text{ and } r\_t < 1-\varepsilon\_{\text{low}} \\\\ 1 & \text{otherwise} \end{cases}
$$

### On-Policy 学習の知見

Global batch = Mini-batch（1回の更新で使い切る）の on-policy 設定では:

- **Importance Sampling 補正が不可欠**: Training-Inference Discrepancy に対する IS 補正を除去すると、エントロピーが急激に低下し学習が崩壊する
- **長さ正規化は性能を劣化させる**: 一次近似の妥当性を壊す（GRPO の $1/|y|$ 正規化への批判）
- **MiniRL が最も安定**: 一次近似の妥当性を保つ目的関数のみが安定かつ高性能を達成

### Off-Policy 学習の知見

Global batch > Mini-batch（複数回の gradient step に分割）の off-policy 設定では:

- **クリッピングと Routing Replay の両方が必要**: どちらか片方だけでは学習崩壊が発生
- **Off-Policiness が小さい場合 (2×)**: R2（学習エンジンの routing を再生）が優位
- **Off-Policiness が大きい場合 (4×–8×)**: R3（推論エンジンの routing を全 mini-batch で再生）が優位
- **Cold-Start の初期化は収束性能に影響しない**: 異なるデータソースで初期化しても、十分な RL 学習の後には同等の性能に収束する

### Off-Policy における正負報酬の非対称性

Off-Policy データにおいて、正の報酬と負の報酬には **非対称的な性質** がある:

- **正の報酬**: Staleness に対して頑健。古いデータでも性能への悪影響が少なく、むしろ改善する場合すらある
- **負の報酬**: Staleness に対して脆弱。3ステップの staleness で精度が 0.73 → 0.48、5ステップで 0.18 まで崩壊

この非対称性は、負の報酬の勾配が「現在のポリシーが探索し始めた領域」の確率を押し下げようとする **非有界な最適化問題** を生むことに起因する。

**非対称クリッピング** がこの問題への解法となる:

- 正の報酬: $c_{\text{pos}} = 10.0$ まで upweighting を許容（学習加速）
- 負の報酬（高 importance ratio）: 1.0 でキャップ（現在好まれる行動の抑制を防止）
- 負の報酬（低 importance ratio）: 0.5 をフロアに設定（勾配シグナルの維持）

---

## 分散 RL フレームワークの設計

RLVR の学習では、**ロールアウト（生成）が学習時間の 80–90% を占める** ため、フレームワーク設計は生成の効率化を中心に構築される。主要フレームワークの設計思想を5つの軸で比較する。

### 5つの設計軸

#### 1. ロールアウトアーキテクチャ: Engine vs Server

| 方式 | 説明 | 採用例 |
|---|---|---|
| **Engine** (Co-located) | 学習と推論を共有メモリ / NCCL で結合。レイテンシは小さいが独立スケーリング不可 | VERL (default), OpenRLHF |
| **Server** (Disaggregated) | HTTP/RPC で分離されたサービス。異種 HW 対応可能 | Slime, VERL (agentic) |

#### 2. 重み同期戦略

| 方式 | 説明 | 採用例 |
|---|---|---|
| **Resharding** | 3D テンソルの device mesh 変換 | VERL |
| **Direct Broadcast** | NCCL / CUDA IPC による単純転送 | OpenRLHF |
| **Versioned Updates** | 非同期キューイング + staleness トラッキング | AReaL |
| **SHARDCAST** | 信頼できないクラスタ間のリレー配信 | PrimeRL |

#### 3. ワーカー構成

- **Monolithic** (VERL): ActorRolloutRefWorker が複数の役割を兼務
- **Role-based** (OpenRLHF): Actor / Critic / Reward を Ray Actor として分離
- **Three-module** (Slime): Trainer / Rollout / Buffer を厳密に分離
- **Async Pool** (AReaL): ロールアウトと学習のワーカーが固定ペアリングなし

#### 4. データフロー

- **DataProto** (VERL): テンソル中心の統一プロトコル
- **Ray Object Store** (OpenRLHF): ゼロコピー参照渡し
- **File-backed Buffer** (Slime): Parquet/JSONL のディスク永続化
- **Streaming Queue** (AReaL): staleness タイムスタンプ付きの無限ストリーム

### 主要フレームワーク

#### VERL

**Hybrid-Controller アーキテクチャ**: 単一の Driver がコントロールフロー、分散 Worker Group が計算を担当。

- **HybridEngine モード**: Actor と Rollout を共有 GPU 上で SPMD 推論。メモリ内での重み resharding により切り替え。ただし tool call 時のバッチ内遅延（head-of-line blocking）が問題
- **AsyncServer モード**: 会話ごとに独立した vLLM/SGLang サーバーを用いるモード。マルチターンの agentic RL に適合

#### OpenRLHF

**Ray ベースのプラグマティックな設計**: Ray + vLLM + DeepSpeed を組み合わせ。

- **Hybrid Engine**: Ray Placement Group 内でコンポーネントを co-locate、NCCL/CUDA IPC で同期
- **AutoTP**: 手動設定なしで GPU 間の Tensor Parallelism を自動構成
- 生成（vLLM）とスコアリング（DeepSpeed forward）の計算特性の違いを認識した非対称設計

#### AReaL (Ant Group)

**完全非同期 RL**: 生成と学習を完全に分離し、ハードウェア利用率を最大化。

- ロールアウトワーカーは継続的にデータを生成し、学習ワーカーはデータが到着次第即座に更新
- **Staleness-aware PPO**: 古すぎるサンプルのフィルタリング、クリッピング範囲の調整、バージョンドリフトに比例した KL 正則化
- **Interruptible Rollouts**: 新しい重みが到着した場合、古いデコードを中断して KV Cache を破棄・再計算
- **同期ベースラインに対して 2.77× の高速化** を達成

#### Slime

**モジュール分離による柔軟性**: Trainer (Megatron-LM) / Rollout (SGLang) / Buffer の3つの独立サービス。

- mbridge による Megatron ↔ HuggingFace 形式の変換
- テンソルフラット化 + バケッティングにより **Qwen3-30B-A3B の重み転送を 8×H100 上で 7秒** に最適化
- `/abort_request` エンドポイントによる部分ロールアウトサポート（tool call 対応）

#### PrimeRL + Verifiers

**共有ファイルシステムベースの協調**: Orchestrator / Inference / Training の3コンポーネントを共有ディレクトリ経由で連携。

- Inference ワーカーは共有ディレクトリからチェックポイントをリロード
- クロスプロバイダ学習のための SHARDCAST（効率的なシャード配信）と TOPLOC（決定論性検証）

#### LlamaRL

**GPU-Direct Memory Access**: Controller がすべてのコンポーネントを orchestrate し、グローバルバリアなしで動作。

- **DDMA (Device-to-Device Memory Access)**: CPU staging なしで GPU 間直接転送（NVLink / InfiniBand）
- **405B スケールで DeepSpeed-Chat に対して 10.7× の高速化**

### フレームワーク比較サマリ

| フレームワーク | ロールアウト | 重み同期 | 非同期レベル | Off-Policy 対応 |
|---|---|---|---|---|
| VERL | Engine / Server | Resharding (NCCL) | 同期 / リクエスト非同期 | 限定的 |
| OpenRLHF | Engine | Direct Broadcast | 同期 | 限定的 |
| AReaL | Async Pool | Versioned Updates | 完全非同期 | Staleness-aware PPO |
| Slime | Server | mbridge + Broadcast | 非同期 | 限定的 |
| PrimeRL | Server | SHARDCAST | 2ステップ非同期 | 明示的補正なし |
| LlamaRL | Engine | DDMA (GPU-Direct) | 同期 | 限定的 |

---

## 今後の方向性

RLVR の発展に伴い、以下のトレンドが明確化しつつある:

1. **重み同期がボトルネック**: フレームワーク設計の大部分が重み同期の効率化に集中。Checkpoint-Engine (MoonshotAI) のような専用ミドルウェアの台頭
2. **非同期化の標準化**: RL レベルの非同期（AReaL）とリクエストレベルの非同期（Slime, VERL server）の両方が普及
3. **系列レベル最適化への移行**: GSPO が示すように、トークンレベルの最適化はノイズが大きく、系列レベルへの移行が MoE モデルの安定化に有効
4. **Off-Policy 手法の成熟**: 正負報酬の非対称性を考慮した適応的なクリッピング・IS 補正の重要性
5. **Agentic RL への対応**: Tool call を含むマルチターン対話への対応が、フレームワーク設計の新たな要件

---

## 参考文献

- Han Fang & Karthik Abinav Sankararaman. *Post-Training 101*. Tokens for Thoughts. [Link](https://tokens-for-thoughts.notion.site/post-training-101)
- Feng Yao. *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*. [Link](https://fengyao.notion.site/off-policy-rl)
- Hanif Leo. *Anatomy of RL Frameworks*. [Link](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- Hanif Leo. *RL Framework Worklog I: Handling Off-Policy*. [Link](https://www.hanifleo.com/rl-framework-worklog-i-handling-off-policy/)
- Chujie Zheng et al. *Group Sequence Policy Optimization*. arXiv:2507.18071, 2025. [Link](https://arxiv.org/abs/2507.18071)
- Chujie Zheng et al. *Stabilizing Reinforcement Learning with LLMs: Formulation and Practices*. arXiv:2512.01374, 2025. [Link](https://arxiv.org/abs/2512.01374)
- DeepSeek-AI. *DeepSeek-R1*. 2025.
- VERL. [GitHub](https://github.com/volcengine/verl)
- OpenRLHF. [GitHub](https://github.com/OpenRLHF/OpenRLHF)
