---
title: "LLM Inference Memory Estimator"
layout: "single"
toolsJS: kv-cache-estimator
ShowReadingTime: false
ShowToc: false
---

Estimate total GPU memory for LLM inference: model weights, KV cache, and activation memory. Supports weight quantization, KV cache quantization, and activation quantization. Fetch model config directly from HuggingFace.

<div class="tool-container" id="kv-cache-tool">

<div class="tool-form">

<div class="tool-hf-fetch">
<div class="tool-input-group">
<label>HuggingFace Model ID</label>
<input type="text" id="kv-hf-model-id" placeholder="e.g. meta-llama/Llama-3.1-8B">
</div>
<button id="kv-hf-fetch-btn" onclick="kvFetchHF()">Fetch</button>
</div>
<div class="tool-hf-fetch-status" id="kv-hf-status"></div>

<div class="tool-input-group tool-input-span-2">
<label>Model Preset</label>
<select id="kv-preset" class="tool-preset-select">
<option value="llama3-8b">Llama-3.1-8B</option>
<option value="llama3-70b">Llama-3.1-70B</option>
<option value="llama3-405b">Llama-3.1-405B</option>
<option value="qwen3-8b">Qwen3-8B</option>
<option value="qwen3-235b-moe">Qwen3-235B-A22B (MoE)</option>
<option value="deepseek-v3">DeepSeek-V3 (MoE)</option>
<option value="custom">Custom</option>
</select>
</div>

<div class="tool-section-label">Model Architecture</div>

<div class="tool-input-group">
<label>Hidden Size (h)</label>
<input type="number" id="kv-hidden-size" value="4096" min="1">
</div>

<div class="tool-input-group">
<label>Intermediate Size (h_ffn, dense)</label>
<input type="number" id="kv-intermediate-size" value="14336" min="1">
</div>

<div class="tool-input-group">
<label>Num Layers (L)</label>
<input type="number" id="kv-num-layers" value="32" min="1">
</div>

<div class="tool-input-group">
<label>Num Attention Heads (a)</label>
<input type="number" id="kv-num-heads" value="32" min="1">
</div>

<div class="tool-input-group">
<label>Num KV Heads (k)</label>
<input type="number" id="kv-num-kv-heads" value="8" min="1">
</div>

<div class="tool-input-group">
<label>Head Dim (d)</label>
<input type="number" id="kv-head-dim" value="128" min="1">
</div>

<div class="tool-input-group">
<label>Vocab Size (v)</label>
<input type="number" id="kv-vocab-size" value="128256" min="1">
</div>

<div class="tool-input-group">
<label>Tie Word Embeddings</label>
<select id="kv-tie-embed">
<option value="0">No (separate LM head)</option>
<option value="1">Yes (shared)</option>
</select>
</div>

<div class="tool-section-label">MoE (set Num Experts = 1 for dense)</div>

<div class="tool-input-group">
<label>Num Routed Experts</label>
<input type="number" id="kv-num-experts" value="1" min="1">
</div>

<div class="tool-input-group">
<label>Num Shared Experts</label>
<input type="number" id="kv-shared-experts" value="0" min="0">
</div>

<div class="tool-input-group">
<label>Expert FFN Size (moe_intermediate_size)</label>
<input type="number" id="kv-expert-ffn" value="14336" min="1">
</div>

<div class="tool-input-group">
<label>Experts per Token (top-k)</label>
<input type="number" id="kv-experts-per-token" value="1" min="1">
</div>

<div class="tool-section-label">Quantization</div>

<div class="tool-input-group">
<label>Weight Precision</label>
<select id="kv-weight-prec">
<option value="2">BF16 / FP16 (2 bytes)</option>
<option value="4">FP32 (4 bytes)</option>
<option value="1">FP8 / INT8 (1 byte)</option>
<option value="0.5">NVFP4 / MXFP4 / INT4 (0.5 bytes)</option>
</select>
</div>

<div class="tool-input-group">
<label>KV Cache Precision</label>
<select id="kv-cache-prec">
<option value="2">FP16 / BF16 (2 bytes)</option>
<option value="1">FP8 / INT8 (1 byte)</option>
<option value="0.5">NVFP4 / MXFP4 / INT4 (0.5 bytes)</option>
</select>
</div>

<div class="tool-input-group">
<label>Activation Precision</label>
<select id="kv-act-prec">
<option value="2">FP16 / BF16 (2 bytes)</option>
<option value="1">FP8 (1 byte)</option>
</select>
</div>

<div class="tool-section-label">Inference Config</div>

<div class="tool-input-group">
<label>Max Sequence Length</label>
<input type="number" id="kv-seq-len" value="4096" min="1">
</div>

<div class="tool-input-group">
<label>Batch Size</label>
<input type="number" id="kv-batch-size" value="1" min="1">
</div>

<div class="tool-input-group">
<label>TP Size (Tensor Parallel)</label>
<input type="number" id="kv-tp-size" value="1" min="1">
</div>

<div class="tool-section-label">GPU</div>

<div class="tool-input-group">
<label>GPU Type</label>
<select id="kv-gpu-type">
<option value="a100-40">A100 (40 GB)</option>
<option value="a100-80">A100 (80 GB)</option>
<option value="h100-80" selected>H100 (80 GB)</option>
<option value="h100-94">H100 NVL (94 GB)</option>
<option value="h200-141">H200 (141 GB)</option>
<option value="b200-180">B200 (180 GB)</option>
</select>
</div>

</div>

<div class="tool-validation" id="kv-validation"></div>

<div class="tool-result" id="kv-result">
<div class="tool-result-title">Memory Breakdown (per GPU)</div>

<div id="kv-arch-info" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--secondary);margin-bottom:12px;"></div>

<div class="tool-bar-chart" id="kv-bar-chart"></div>

<table class="tool-result-table" id="kv-main-table">
<thead>
<tr>
<th>Component</th>
<th>Memory</th>
<th>Formula</th>
</tr>
</thead>
<tbody>
<tr>
<td>Model Weights</td>
<td id="kv-r-weights">—</td>
<td class="tool-formula-cell" id="kv-f-weights"></td>
</tr>
<tr>
<td>KV Cache</td>
<td id="kv-r-kvcache">—</td>
<td class="tool-formula-cell" id="kv-f-kvcache"></td>
</tr>
<tr>
<td>Activation Memory (1 layer peak)</td>
<td id="kv-r-act">—</td>
<td class="tool-formula-cell" id="kv-f-act"></td>
</tr>
<tr class="tool-total-row">
<td>Total</td>
<td id="kv-r-total">—</td>
<td></td>
</tr>
</tbody>
</table>

<div class="tool-formula" id="kv-param-summary">
<div class="tool-formula-label">Model Parameters</div>
<code id="kv-param-detail">—</code>
</div>

<div class="tool-formula" id="kv-cache-summary">
<div class="tool-formula-label">KV Cache Formula</div>
<code id="kv-cache-detail">—</code>
</div>

<div class="tool-gpu-fit" id="kv-gpu-fit">
<div class="tool-gpu-fit-title">GPU Fit Check</div>
<div class="tool-gpu-fit-cards" id="kv-gpu-cards"></div>
</div>
</div>

<div class="tool-result" id="kv-roofline-result">
<div class="tool-result-title">Roofline Analysis (per layer, decode)</div>
<p style="font-size:12px;color:var(--secondary);margin-bottom:10px;">
Arithmetic Intensity (AI) = FLOPs / Bytes. If AI &lt; ops:byte ratio of the GPU, the operation is <span class="tool-bound-tag memory">Memory Bound</span>; otherwise <span class="tool-bound-tag compute">Compute Bound</span>.
</p>
<table class="tool-op-table" id="kv-op-table">
<thead>
<tr>
<th>Operation</th>
<th>FLOPs</th>
<th>Bytes</th>
<th>AI (F/B)</th>
<th>Bound</th>
</tr>
</thead>
<tbody id="kv-op-tbody"></tbody>
</table>
<div class="tool-formula" style="margin-top:12px">
<div class="tool-formula-label">References</div>
<code style="font-size:11px;word-break:break-word;">
<a href="https://arxiv.org/abs/2411.06465" style="color:var(--accent)">Fujii et al., &ldquo;Accelerating Large Language Model Training with 4D Parallelism and Memory Consumption Estimator&rdquo; (arXiv:2411.06465)</a> &mdash; by the author of this tool &bull;
Williams et al., &ldquo;Roofline: An Insightful Visual Performance Model&rdquo; (2009) &bull;
<a href="https://jax-ml.github.io/scaling-book/roofline/" style="color:var(--accent)">JAX Scaling Book &mdash; Rooflines</a> &bull;
<a href="https://kipp.ly/transformer-inference-arithmetic/" style="color:var(--accent)">kipply &mdash; Transformer Inference Arithmetic</a>
</code>
</div>
</div>

<div class="tool-note">
<div class="tool-note-title">Note</div>
This estimator targets <strong>Transformer-based LLMs</strong> (decoder-only, with SwiGLU FFN and RMSNorm). It does not support SSM-based models (Mamba, RWKV, etc.), Gated Linear Networks, or Diffusion models. Additionally:
<ul style="margin:8px 0 0 16px;font-size:13px;color:var(--secondary)">
<li>KV cache assumes the full max sequence length is allocated. With <strong>paged attention</strong> (e.g., vLLM, TGI), pages are allocated on-demand, so actual usage may be lower.</li>
<li>Inference frameworks have their own memory overhead (CUDA context ~300&ndash;500 MB, framework buffers, etc.) which is not included here.</li>
</ul>
</div>

</div>

<div class="tool-formula" style="margin-top:16px">
<div class="tool-formula-label">Citation</div>
<code style="font-size:11px;word-break:break-word;">@misc{fujii2024acceleratinglargelanguagemodel,
      title={Accelerating Large Language Model Training with 4D Parallelism and Memory Consumption Estimator},
      author={Kazuki Fujii and Kohei Watanabe and Rio Yokota},
      year={2024},
      eprint={2411.06465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06465},
}</code>
</div>
