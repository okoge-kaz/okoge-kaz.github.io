---
title: "Tools"
description: "Interactive estimators for LLM practitioners — inference memory, training memory, GPU planning, and more."
---

## Interactive Estimators

Practical client-side tools for planning LLM training and inference. All calculations run entirely in your browser. Fetch model configs directly from HuggingFace.

<div class="tool-cards">

<a class="tool-card" href="/tools/kv-cache-estimator/">
<div class="tool-card-title">LLM Inference Memory Estimator</div>
<div class="tool-card-desc">Estimate GPU memory for LLM inference: model weights, KV cache, and activations. Supports weight/KV/activation quantization and roofline analysis.</div>
</a>

<a class="tool-card" href="/tools/memory-estimator/">
<div class="tool-card-title">LLM Training Memory Estimator</div>
<div class="tool-card-desc">Estimate per-GPU memory during LLM training with 5D parallelism (TP/PP/DP/CP/EP). Supports MoE, distributed optimizer, and NCCL buffer estimation.</div>
</a>

<a class="tool-card" href="/tools/gpu-process-visualizer/">
<div class="tool-card-title">GPU Process Mapping Visualizer</div>
<div class="tool-card-desc">Visualize how TP/EP/CP/DP/PP communicator groups map to physical GPU nodes. Interactive 3D view with color-coded process groups.</div>
</a>

</div>
