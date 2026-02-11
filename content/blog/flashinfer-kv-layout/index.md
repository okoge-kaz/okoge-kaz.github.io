---
title: "Understanding FlashInfer's KV-Cache Layouts"
draft: true
date: 2026-02-08
tags: ["LLM", "FlashInfer", "KV-Cache", "Inference", "CUDA"]
description: "A deep dive into FlashInfer's KV-Cache layout designs — Ragged Tensor, Paged KV-Cache, and Cascade Inference — and how they enable efficient LLM serving."
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---

## Introduction

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is a high-performance GPU kernel library for LLM inference. vLLM, SGLang, TensorRT-LLM などの主要な推論フレームワークのバックエンドとして採用されている。

本記事では FlashInfer の KV-Cache Layout について解説する。KV-Cache のメモリレイアウトは推論性能に直結する重要な設計要素であり、FlashInfer がどのようなデータ構造を提供しているかを理解することは、LLM 推論の最適化を行う上で不可欠である。

公式ドキュメント: [KV-Cache Layout in FlashInfer](https://docs.flashinfer.ai/tutorials/kv_layout.html)

## NHD vs HND Layout

FlashInfer では KV-Cache の最後の3次元の並び順として **NHD** と **HND** の2種類をサポートしている。

- **NHD**: `(seq_len, num_heads, head_dim)` — Projection 演算 $xW_K$, $xW_V$ の出力と一致する自然な配置。FlashInfer のデフォルト。
- **HND**: `(num_heads, seq_len, head_dim)` — FP8 などの低精度フォーマットで GPU 効率が向上する。FP16 では NHD との性能差はほぼない。

<!-- TODO: NHD/HND の性能比較ベンチマーク結果を追加 -->

## Ragged Tensor

可変長シーケンスのバッチ処理（Batch Prefill）において、パディングなしでリクエストを連結するためのデータ構造。

### 構造

- **data tensor**: 全リクエストの Q/K/V を連結した1次元テンソル。形状は `(total_tokens, num_heads, head_dim)` (NHD の場合)
- **indptr array**: 長さ `num_requests + 1` の整数配列。リクエスト $i$ のトークンは `data[indptr[i]:indptr[i+1]]` でアクセスできる。

```
Example: 3 requests with lengths [3, 1, 2]

indptr = [0, 3, 4, 6]

data:  [t0, t1, t2 | t3 | t4, t5]
        ← req 0 →  ←r1→ ← req 2→
```

> **注意**: `indptr` 配列は `int32` 型でなければならない。`int64` を使用するとインデックスエラーが発生する。

対応 API: `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`

<!-- TODO: 実際のコード例を追加 -->

## Mask Layout (2D Ragged Tensor)

複数リクエストのシーケンス長が異なる場合の Attention Mask を、2D Ragged Tensor として表現する。

- **qo_indptr**: リクエストごとの Query 長を追跡
- **kv_indptr**: リクエストごとの KV 長を追跡
- **mask_data**: 全リクエストの Mask を連結した1次元配列

Mask の値は Softmax の前（Scaling の後）に Attention Score に加算される。

### Bit-Packing

Boolean mask を bit-packed 形式に圧縮可能（1要素あたり1ビット、8要素を1つの uint8 に格納）。

```python
flashinfer.quantization.packbits()
flashinfer.quantization.segment_packbits()
```

<!-- TODO: Causal mask との関係、具体的な使用例 -->

## Paged KV-Cache

vLLM スタイルの Paged KV-Cache を **Compressed Sparse Row (CSR)** 形式で実装している。

### Page Table

各リクエストは以下で管理される:

- `page_indices`: 割り当てられたページのインデックス列
- `last_page_len`: 最終ページに格納されたエントリ数

シーケンス長は次の式で計算できる:

$$\text{seq\_len} = \text{page\_size} \times (\text{len(page\_indices)} - 1) + \text{last\_page\_len}$$

> **制約**: `last_page_len` は $0 < \text{last\_page\_len} \leq \text{page\_size}$ を満たす必要がある。

### ストレージ形式

**5次元テンソル形式**（K と V を1つのテンソルで管理）:
```
NHD: (max_num_pages, 2, page_size, num_heads, head_dim)
HND: (max_num_pages, 2, num_heads, page_size, head_dim)
```

**タプル形式**（K と V を別々のテンソルで管理）:
```
k_data: (max_num_pages, page_size, num_heads, head_dim)  # NHD
v_data: (max_num_pages, page_size, num_heads, head_dim)  # NHD
```

対応 API:
- `flashinfer.page.append_paged_kv_cache()`
- `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper`
- `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper`

<!-- TODO: CSR 形式の図解、vLLM の PagedAttention との比較 -->

## Multi-Head Latent Attention (MLA) Page Layout

DeepSeek v2 の MLA では K と V のキャッシュを分離せず統合して管理する。NHD/HND の区別も不要。

- **ckv** (compressed key/value): メインのキャッシュ
- **kpe** (key with positional encoding): RoPE 次元

これらは1つのテンソル内で適切なスライスにより共存でき、データ移動なしでアクセス可能。

対応 API: `flashinfer.mla.BatchMLAPagedAttentionWrapper`

<!-- TODO: MLA の仕組みと KV Cache 圧縮の詳細 -->

## Multi-Level Cascade Inference

Prefix 共有推論（System Prompt の共有など）のための多段構造。

- Ragged な Query/Output は全レベルで共有
- Paged KV-Cache も全レベルで統一
- レベルごとに独立したインデックス配列を持つ:
  - `qo_indptr`, `kv_page_indptr`, `kv_page_indices`, `kv_last_page_len`

これにより、同一の物理データに対して異なる論理ビューを提供できる。

対応 API: `flashinfer.cascade.MultiLevelCascadeAttentionWrapper`

<!-- TODO: Cascade Inference の具体的なユースケースと性能評価 -->

## まとめ

<!-- TODO: 各レイアウトの使い分けガイドラインを追加 -->

FlashInfer は KV-Cache の管理を推論フレームワーク側に委ねつつ、Attention 計算に特化した高効率カーネルを提供する設計となっている。Page Table の管理（ページの割り当て・解放）は FlashInfer の責務ではなく、ユーザー側のフレームワークに任されている。

---

## References

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer KV-Cache Layout Documentation](https://docs.flashinfer.ai/tutorials/kv_layout.html)
- [vLLM PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)
