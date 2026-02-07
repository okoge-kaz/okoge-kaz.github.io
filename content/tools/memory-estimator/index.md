---
title: "LLM Training Memory Estimator"
layout: "single"
toolsJS: memory-estimator
ShowReadingTime: false
ShowToc: false
---

Estimate per-GPU memory consumption during LLM training with 5D parallelism (TP, PP, DP, CP, EP). Supports MoE, distributed optimizer, and NCCL buffer estimation. Fetch model config from HuggingFace.

<div class="tool-container" id="mem-tool">

<div class="tool-form">

<div class="tool-hf-fetch">
<div class="tool-input-group">
<label>HuggingFace Model ID</label>
<input type="text" id="mem-hf-model-id" placeholder="e.g. meta-llama/Llama-3.1-8B">
</div>
<button id="mem-hf-fetch-btn" onclick="memFetchHF()">Fetch</button>
</div>
<div class="tool-hf-fetch-status" id="mem-hf-status"></div>

<div class="tool-input-group tool-input-span-2">
<label>Model Preset</label>
<select id="mem-preset" class="tool-preset-select">
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
<input type="number" id="mem-hidden-size" value="4096" min="1">
</div>

<div class="tool-input-group">
<label>Intermediate Size (h_ffn, dense)</label>
<input type="number" id="mem-intermediate-size" value="14336" min="1">
</div>

<div class="tool-input-group">
<label>Num Layers (L)</label>
<input type="number" id="mem-num-layers" value="32" min="1">
</div>

<div class="tool-input-group">
<label>Num Attention Heads (a)</label>
<input type="number" id="mem-num-heads" value="32" min="1">
</div>

<div class="tool-input-group">
<label>Num KV Heads (k)</label>
<input type="number" id="mem-num-kv-heads" value="8" min="1">
</div>

<div class="tool-input-group">
<label>Head Dim (d)</label>
<input type="number" id="mem-head-dim" value="128" min="1">
</div>

<div class="tool-input-group">
<label>Vocab Size (v)</label>
<input type="number" id="mem-vocab-size" value="128256" min="1">
</div>

<div class="tool-input-group">
<label>Tie Word Embeddings</label>
<select id="mem-tie-embed">
<option value="0">No (separate LM head)</option>
<option value="1">Yes (shared)</option>
</select>
</div>

<div class="tool-section-label">MoE (set Num Experts = 1 for dense)</div>

<div class="tool-input-group">
<label>Num Routed Experts</label>
<input type="number" id="mem-num-experts" value="1" min="1">
</div>

<div class="tool-input-group">
<label>Num Shared Experts</label>
<input type="number" id="mem-shared-experts" value="0" min="0">
</div>

<div class="tool-input-group">
<label>Expert FFN Size (moe_intermediate_size)</label>
<input type="number" id="mem-expert-ffn" value="14336" min="1">
</div>

<div class="tool-input-group">
<label>Experts per Token (top-k)</label>
<input type="number" id="mem-experts-per-token" value="1" min="1">
</div>

<div class="tool-section-label">Precision &amp; Optimizer</div>

<div class="tool-input-group">
<label>Param Dtype</label>
<select id="mem-param-dtype">
<option value="2">BF16 (2 bytes)</option>
<option value="4">FP32 (4 bytes)</option>
<option value="1">FP8 (1 byte)</option>
<option value="0.5">NVFP4 / MXFP4 (0.5 bytes)</option>
</select>
</div>

<div class="tool-input-group">
<label>Grad Dtype</label>
<select id="mem-grad-dtype">
<option value="2">BF16 (2 bytes)</option>
<option value="4">FP32 (4 bytes)</option>
</select>
</div>

<div class="tool-input-group">
<label>Optimizer</label>
<select id="mem-optimizer">
<option value="adam">Adam (master FP32 + m + v = 12 B/param)</option>
<option value="sgd">SGD (master FP32 + m = 8 B/param)</option>
</select>
</div>

<div class="tool-section-label">Training Config</div>

<div class="tool-input-group">
<label>Sequence Length (s)</label>
<input type="number" id="mem-seq-len" value="4096" min="1">
</div>

<div class="tool-input-group">
<label>Micro Batch Size (b)</label>
<input type="number" id="mem-micro-batch" value="1" min="1">
</div>

<div class="tool-section-label">Parallelism</div>

<div class="tool-input-group">
<label>TP Size (Tensor Parallel)</label>
<input type="number" id="mem-tp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>PP Size (Pipeline Parallel)</label>
<input type="number" id="mem-pp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>DP Size (Data Parallel)</label>
<input type="number" id="mem-dp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>CP Size (Context Parallel)</label>
<input type="number" id="mem-cp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>EP Size (Expert Parallel)</label>
<input type="number" id="mem-ep" value="1" min="1">
</div>

<div class="tool-input-group">
<label>Distributed Optimizer</label>
<select id="mem-dist-optim">
<option value="1">ON (shard across DP)</option>
<option value="0">OFF</option>
</select>
</div>

<div class="tool-input-group">
<label>PP Scheduler</label>
<select id="mem-pp-sched">
<option value="1f1b">1F1B</option>
<option value="interleaved">Interleaved 1F1B</option>
</select>
</div>

<div class="tool-input-group">
<label>Virtual PP Stages</label>
<input type="number" id="mem-vpp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>Standalone Embedding Stage</label>
<select id="mem-standalone-embed">
<option value="0">OFF (embed in first/last PP stage)</option>
<option value="1">ON (embed/LM head as separate PP stages)</option>
</select>
</div>

<div class="tool-section-label">GPU</div>

<div class="tool-input-group">
<label>GPU Type</label>
<select id="mem-gpu-type">
<option value="a100-40">A100 (40 GB)</option>
<option value="a100-80">A100 (80 GB)</option>
<option value="h100-80" selected>H100 (80 GB)</option>
<option value="h100-94">H100 NVL (94 GB)</option>
<option value="h200-141">H200 (141 GB)</option>
<option value="b200-180">B200 (180 GB)</option>
</select>
</div>

</div>

<div class="tool-validation" id="mem-validation"></div>

<div class="tool-result" id="mem-result">
<div class="tool-result-title">Memory Breakdown (per GPU)</div>

<div id="mem-arch-info" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--secondary);margin-bottom:12px;"></div>

<div class="tool-bar-chart" id="mem-bar-chart"></div>

<table class="tool-result-table" id="mem-table">
<thead>
<tr>
<th>Component</th>
<th>Memory</th>
<th>Formula</th>
</tr>
</thead>
<tbody>
<tr>
<td>Parameters</td>
<td id="mem-r-params">—</td>
<td class="tool-formula-cell" id="mem-f-params"></td>
</tr>
<tr>
<td>Gradients</td>
<td id="mem-r-grad">—</td>
<td class="tool-formula-cell" id="mem-f-grad"></td>
</tr>
<tr>
<td>Optimizer States</td>
<td id="mem-r-optim">—</td>
<td class="tool-formula-cell" id="mem-f-optim"></td>
</tr>
<tr>
<td>Activations</td>
<td id="mem-r-act">—</td>
<td class="tool-formula-cell" id="mem-f-act"></td>
</tr>
<tr>
<td>NCCL Buffers (est.)</td>
<td id="mem-r-nccl">—</td>
<td class="tool-formula-cell" id="mem-f-nccl"></td>
</tr>
<tr class="tool-total-row">
<td>Total</td>
<td id="mem-r-total">—</td>
<td></td>
</tr>
</tbody>
</table>

<div class="tool-formula" id="mem-param-summary">
<div class="tool-formula-label">Model Parameters</div>
<code id="mem-param-detail">—</code>
</div>

<div class="tool-gpu-fit" id="mem-gpu-fit">
<div class="tool-gpu-fit-title">GPU Fit Check</div>
<div class="tool-gpu-fit-cards" id="mem-gpu-cards"></div>
</div>
</div>

<div class="tool-result" id="mem-roofline-result">
<div class="tool-result-title">Roofline Analysis (Training, per layer)</div>
<p style="font-size:12px;color:var(--secondary);margin-bottom:10px;">
Arithmetic Intensity (AI) = FLOPs / Bytes. If AI &lt; ops:byte ratio of the GPU, the operation is <span class="tool-bound-tag memory">Memory Bound</span>; otherwise <span class="tool-bound-tag compute">Compute Bound</span>.
</p>
<table class="tool-op-table" id="mem-op-table">
<thead>
<tr>
<th>Operation</th>
<th>FLOPs</th>
<th>Bytes</th>
<th>AI (F/B)</th>
<th>Bound</th>
</tr>
</thead>
<tbody id="mem-op-tbody"></tbody>
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
This estimator targets <strong>Transformer-based LLMs</strong> (decoder-only, with SwiGLU FFN and RMSNorm). It does not support SSM-based models (Mamba, RWKV, etc.), Gated Linear Networks, or Diffusion models. These are <strong>theoretical estimates</strong>. Actual memory consumption during training includes additional overheads not modeled here:
<ul style="margin:8px 0 0 16px;font-size:13px;color:var(--secondary)">
<li><strong>NCCL communication buffers</strong>: Each parallelism dimension creates its own communicator. Total NCCL overhead can reach 1&ndash;20 GB depending on the number of communicators and <code>NCCL_BUFFSIZE</code>. This estimator includes a rough estimate (~0.5 GB per communicator).</li>
<li><strong>CUDA context</strong>: ~300&ndash;500 MB per GPU for the CUDA runtime and driver.</li>
<li><strong>Memory fragmentation</strong>: PyTorch's caching allocator may hold more memory than actually used.</li>
<li><strong>Temporary buffers</strong>: All-gather buffers during distributed optimizer gather, pipeline send/recv buffers, etc.</li>
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
