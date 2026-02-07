(function () {
  'use strict';

  // ===== GPU Specs =====
  var GPUS = {
    'a100-40':  { name: 'A100 40GB',  mem: 40e9,  bf16Tflops: 312,  bwTBs: 2.0   },
    'a100-80':  { name: 'A100 80GB',  mem: 80e9,  bf16Tflops: 312,  bwTBs: 2.0   },
    'h100-80':  { name: 'H100 80GB',  mem: 80e9,  bf16Tflops: 989,  bwTBs: 3.35  },
    'h100-94':  { name: 'H100 NVL',   mem: 94e9,  bf16Tflops: 989,  bwTBs: 3.35  },
    'h200-141': { name: 'H200 141GB', mem: 141e9, bf16Tflops: 989,  bwTBs: 4.8   },
    'b200-180': { name: 'B200 180GB', mem: 180e9, bf16Tflops: 2250, bwTBs: 8.0   },
  };

  // ===== Model Presets =====
  // h_ffn = intermediate_size (dense layers), expertHffn = moe_intermediate_size (per-expert)
  var PRESETS = {
    'llama3-8b':      { h: 4096,  h_ffn: 14336, L: 32,  a: 32,  k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 14336, topk: 1, seqLen: 8192   },
    'llama3-70b':     { h: 8192,  h_ffn: 28672, L: 80,  a: 64,  k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 28672, topk: 1, seqLen: 8192   },
    'llama3-405b':    { h: 16384, h_ffn: 53248, L: 126, a: 128, k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 53248, topk: 1, seqLen: 131072 },
    'qwen3-8b':       { h: 4096,  h_ffn: 12288, L: 36,  a: 32,  k: 8,   headDim: 128, v: 151936, tiedEmbed: true,  experts: 1,   sharedExperts: 0, expertHffn: 12288, topk: 1, seqLen: 32768  },
    'qwen3-72b':      { h: 8192,  h_ffn: 29568, L: 80,  a: 64,  k: 8,   headDim: 128, v: 152064, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 29568, topk: 1, seqLen: 32768  },
    'qwen3-235b-moe': { h: 4096,  h_ffn: 12288, L: 94,  a: 64,  k: 4,   headDim: 64,  v: 151936, tiedEmbed: true,  experts: 128, sharedExperts: 0, expertHffn: 1536,  topk: 8, seqLen: 32768  },
    'deepseek-v3':    { h: 7168,  h_ffn: 18432, L: 61,  a: 128, k: 128, headDim: 128, v: 129280, tiedEmbed: false, experts: 256, sharedExperts: 1, expertHffn: 2048,  topk: 8, seqLen: 65536  },
  };

  var els = {};

  function $(id) { return document.getElementById(id); }

  function init() {
    els.preset       = $('kv-preset');
    els.hiddenSize   = $('kv-hidden-size');
    els.interSize    = $('kv-intermediate-size');
    els.layers       = $('kv-num-layers');
    els.numHeads     = $('kv-num-heads');
    els.kvHeads      = $('kv-num-kv-heads');
    els.headDim      = $('kv-head-dim');
    els.vocabSize    = $('kv-vocab-size');
    els.numExperts   = $('kv-num-experts');
    els.sharedExperts= $('kv-shared-experts');
    els.expertFfn    = $('kv-expert-ffn');
    els.topK         = $('kv-experts-per-token');
    els.tieEmbed     = $('kv-tie-embed');
    els.weightPrec   = $('kv-weight-prec');
    els.cachePrec    = $('kv-cache-prec');
    els.actPrec      = $('kv-act-prec');
    els.seqLen       = $('kv-seq-len');
    els.batchSize    = $('kv-batch-size');
    els.tpSize       = $('kv-tp-size');
    els.gpuType      = $('kv-gpu-type');

    els.preset.addEventListener('change', applyPreset);

    var inputs = document.querySelectorAll('#kv-cache-tool input, #kv-cache-tool select');
    for (var i = 0; i < inputs.length; i++) {
      inputs[i].addEventListener('input', calculate);
      inputs[i].addEventListener('change', calculate);
    }

    applyPreset();
  }

  function applyPreset() {
    var key = els.preset.value;
    var p = PRESETS[key];
    if (!p) return calculate();
    els.hiddenSize.value   = p.h;
    els.interSize.value    = p.h_ffn;
    els.layers.value       = p.L;
    els.numHeads.value     = p.a;
    els.kvHeads.value      = p.k;
    els.headDim.value      = p.headDim;
    els.vocabSize.value    = p.v;
    els.numExperts.value   = p.experts;
    els.sharedExperts.value= p.sharedExperts;
    els.expertFfn.value    = p.expertHffn;
    els.topK.value         = p.topk;
    els.tieEmbed.value     = p.tiedEmbed ? '1' : '0';
    els.seqLen.value       = p.seqLen;
    calculate();
  }

  // ===== HuggingFace Config Fetch =====
  window.kvFetchHF = function () {
    var modelId = $('kv-hf-model-id').value.trim();
    var statusEl = $('kv-hf-status');
    if (!modelId) {
      statusEl.textContent = 'Please enter a model ID';
      statusEl.className = 'tool-hf-fetch-status error';
      return;
    }
    statusEl.textContent = 'Fetching config.json...';
    statusEl.className = 'tool-hf-fetch-status';

    var url = 'https://huggingface.co/' + modelId + '/resolve/main/config.json';
    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (cfg) {
        if (cfg.hidden_size) els.hiddenSize.value = cfg.hidden_size;
        if (cfg.intermediate_size) els.interSize.value = cfg.intermediate_size;
        if (cfg.num_hidden_layers) els.layers.value = cfg.num_hidden_layers;
        if (cfg.num_attention_heads) els.numHeads.value = cfg.num_attention_heads;
        if (cfg.num_key_value_heads) {
          els.kvHeads.value = cfg.num_key_value_heads;
        } else if (cfg.num_attention_heads) {
          els.kvHeads.value = cfg.num_attention_heads;
        }
        // Head dim: explicit field or computed
        if (cfg.head_dim) {
          els.headDim.value = cfg.head_dim;
        } else if (cfg.v_head_dim) {
          // DeepSeek MLA: use v_head_dim for KV cache sizing
          els.headDim.value = cfg.v_head_dim;
        } else if (cfg.hidden_size && cfg.num_attention_heads) {
          els.headDim.value = Math.floor(cfg.hidden_size / cfg.num_attention_heads);
        }
        if (cfg.vocab_size) els.vocabSize.value = cfg.vocab_size;

        // MoE fields
        var nExperts = cfg.n_routed_experts || cfg.num_local_experts || cfg.num_experts || 1;
        els.numExperts.value = nExperts;
        els.sharedExperts.value = cfg.n_shared_experts || 0;

        // Expert FFN size: moe_intermediate_size or fall back to intermediate_size
        if (cfg.moe_intermediate_size) {
          els.expertFfn.value = cfg.moe_intermediate_size;
        } else {
          els.expertFfn.value = cfg.intermediate_size || els.interSize.value;
        }

        if (cfg.num_experts_per_tok) {
          els.topK.value = cfg.num_experts_per_tok;
        } else if (cfg.num_selected_experts) {
          els.topK.value = cfg.num_selected_experts;
        } else {
          els.topK.value = nExperts > 1 ? 1 : 1;
        }

        // tie_word_embeddings: default is true in HF if not specified
        if (cfg.tie_word_embeddings === false) {
          els.tieEmbed.value = '0';
        } else {
          els.tieEmbed.value = '1';
        }

        if (cfg.max_position_embeddings) els.seqLen.value = cfg.max_position_embeddings;

        els.preset.value = 'custom';
        statusEl.textContent = 'Loaded: ' + modelId;
        statusEl.className = 'tool-hf-fetch-status success';
        calculate();
      })
      .catch(function (e) {
        statusEl.textContent = 'Error: ' + e.message + '. Check model ID or try a public model.';
        statusEl.className = 'tool-hf-fetch-status error';
      });
  };

  function formatBytes(bytes) {
    if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
    if (bytes >= 1e9)  return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6)  return (bytes / 1e6).toFixed(2) + ' MB';
    if (bytes >= 1e3)  return (bytes / 1e3).toFixed(2) + ' KB';
    return bytes.toFixed(0) + ' B';
  }

  function formatNum(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9)  return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6)  return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3)  return (n / 1e3).toFixed(2) + 'K';
    return n.toString();
  }

  function formatFlops(f) {
    if (f >= 1e15) return (f / 1e15).toFixed(2) + ' PFLOPs';
    if (f >= 1e12) return (f / 1e12).toFixed(2) + ' TFLOPs';
    if (f >= 1e9)  return (f / 1e9).toFixed(2) + ' GFLOPs';
    if (f >= 1e6)  return (f / 1e6).toFixed(2) + ' MFLOPs';
    return f.toFixed(0) + ' FLOPs';
  }

  function getAttnType(a, k) {
    if (k === 1) return 'MQA';
    if (k === a) return 'MHA';
    return 'GQA (ratio ' + (a / k) + ':1)';
  }

  function calculate() {
    var h        = parseInt(els.hiddenSize.value) || 0;
    var h_ffn    = parseInt(els.interSize.value) || 0;
    var L        = parseInt(els.layers.value) || 0;
    var a        = parseInt(els.numHeads.value) || 0;
    var k        = parseInt(els.kvHeads.value) || 0;
    var headDim  = parseInt(els.headDim.value) || 0;
    var v        = parseInt(els.vocabSize.value) || 0;
    var numExperts    = parseInt(els.numExperts.value) || 1;
    var sharedExperts = parseInt(els.sharedExperts.value) || 0;
    var expertHffn    = parseInt(els.expertFfn.value) || h_ffn;
    var topK     = parseInt(els.topK.value) || 1;
    var tiedEmbed = parseInt(els.tieEmbed.value) === 1;
    var wBytes   = parseFloat(els.weightPrec.value) || 2;
    var kvBytes  = parseFloat(els.cachePrec.value) || 2;
    var actBytes = parseFloat(els.actPrec.value) || 2;
    var s        = parseInt(els.seqLen.value) || 0;
    var b        = parseInt(els.batchSize.value) || 1;
    var tp       = parseInt(els.tpSize.value) || 1;
    var gpuKey   = els.gpuType.value;

    var isMoE = numExperts > 1;

    // ===== Architecture Info =====
    var archInfo = getAttnType(a, k);
    if (isMoE) archInfo += ' | MoE (' + numExperts + ' routed + ' + sharedExperts + ' shared, top-' + topK + ', expert_ffn=' + expertHffn + ')';
    else archInfo += ' | Dense FFN';
    archInfo += ' | head_dim=' + headDim;
    $('kv-arch-info').textContent = archInfo;

    // ===== Parameter Count =====
    // Attention per layer: Q(h, a*headDim) + K(h, k*headDim) + V(h, k*headDim) + O(a*headDim, h)
    var attnParamsPerLayer = h * a * headDim + h * k * headDim + h * k * headDim + a * headDim * h;
    // = 2 * h * a * headDim + 2 * h * k * headDim (when a*headDim == h, this simplifies)

    // FFN per layer
    var ffnParamsPerLayer;
    if (isMoE) {
      // Routed experts: each has 3 projections of size h * expertHffn
      // Shared experts: same architecture but always active
      // Router gate: h * numExperts
      ffnParamsPerLayer = numExperts * 3 * h * expertHffn
                        + sharedExperts * 3 * h * expertHffn
                        + h * numExperts; // router
    } else {
      ffnParamsPerLayer = 3 * h * h_ffn;
    }

    // RMSNorm: 2 per layer
    var normParamsPerLayer = 2 * h;

    var paramsPerLayer = attnParamsPerLayer + ffnParamsPerLayer + normParamsPerLayer;
    var totalLayerParams = paramsPerLayer * L;

    // Embedding + LM head + final norm
    var embedParams = h * v;
    var lmHeadParams = tiedEmbed ? 0 : h * v;
    var finalNormParams = h;
    var totalParams = totalLayerParams + embedParams + lmHeadParams + finalNormParams;

    // Per GPU (TP splits attention and FFN, norm replicated)
    var totalParamsPerGpu = (attnParamsPerLayer * L + ffnParamsPerLayer * L) / tp
                          + normParamsPerLayer * L
                          + (embedParams + lmHeadParams) / tp + finalNormParams;

    // ===== Model Weight Memory =====
    var weightMem = totalParamsPerGpu * wBytes;

    // ===== KV Cache Memory =====
    // 2 (K+V) * L * k * headDim * s * b * kvBytes / tp
    var kvCacheMem = 2 * L * k * headDim * s * b * kvBytes / tp;

    // ===== Activation Memory (inference) =====
    // Only 1 layer's activations live at a time
    // Peak: FFN intermediate is the largest tensor
    var actFfnDim = isMoE ? expertHffn * topK : h_ffn;
    var actMem = s * b * (2 * h + 2 * actFfnDim) * actBytes / tp;

    // ===== Total =====
    var totalMem = weightMem + kvCacheMem + actMem;

    // ===== Update Results =====
    $('kv-r-weights').textContent = formatBytes(weightMem);
    $('kv-f-weights').textContent = formatNum(totalParamsPerGpu) + ' params \u00D7 ' + wBytes + 'B = ' + formatBytes(weightMem);

    $('kv-r-kvcache').textContent = formatBytes(kvCacheMem);
    $('kv-f-kvcache').textContent = '2 \u00D7 ' + L + ' \u00D7 ' + k + ' \u00D7 ' + headDim + ' \u00D7 ' + s + ' \u00D7 ' + b + ' \u00D7 ' + kvBytes + 'B / ' + tp;

    $('kv-r-act').textContent = formatBytes(actMem);
    $('kv-f-act').textContent = '1 layer peak: s\u00D7b\u00D7(2h + 2\u00D7' + (isMoE ? 'topk\u00D7expert_ffn' : 'h_ffn') + ') \u00D7 ' + actBytes + 'B / tp';

    $('kv-r-total').textContent = formatBytes(totalMem);

    // Parameter summary — detailed breakdown
    var pdHtml = '';
    pdHtml += '<b>Attention per layer:</b> Q(h\u2192a\u00B7d) + K(h\u2192k\u00B7d) + V(h\u2192k\u00B7d) + O(a\u00B7d\u2192h)\n';
    pdHtml += '  = ' + h + '\u00D7' + (a * headDim) + ' + ' + h + '\u00D7' + (k * headDim) + ' + ' + h + '\u00D7' + (k * headDim) + ' + ' + (a * headDim) + '\u00D7' + h + ' = <b>' + formatNum(attnParamsPerLayer) + '</b>\n';
    if (isMoE) {
      pdHtml += '<b>FFN per layer (MoE):</b>\n';
      pdHtml += '  Routed experts: ' + numExperts + ' \u00D7 3 \u00D7 ' + h + ' \u00D7 ' + expertHffn + ' = ' + formatNum(numExperts * 3 * h * expertHffn) + '\n';
      if (sharedExperts > 0) {
        pdHtml += '  Shared experts: ' + sharedExperts + ' \u00D7 3 \u00D7 ' + h + ' \u00D7 ' + expertHffn + ' = ' + formatNum(sharedExperts * 3 * h * expertHffn) + '\n';
      }
      pdHtml += '  Router gate:    ' + h + ' \u00D7 ' + numExperts + ' = ' + formatNum(h * numExperts) + '\n';
      pdHtml += '  FFN total/layer = <b>' + formatNum(ffnParamsPerLayer) + '</b>\n';
    } else {
      pdHtml += '<b>FFN per layer (SwiGLU):</b> 3 \u00D7 ' + h + ' \u00D7 ' + h_ffn + ' = <b>' + formatNum(ffnParamsPerLayer) + '</b>\n';
    }
    pdHtml += '<b>Norm per layer:</b> 2 \u00D7 ' + h + ' = ' + formatNum(normParamsPerLayer) + '\n';
    pdHtml += '<b>Per layer total:</b> ' + formatNum(attnParamsPerLayer + ffnParamsPerLayer + normParamsPerLayer) + '\n';
    pdHtml += '<b>All ' + L + ' layers:</b> ' + formatNum(attnParamsPerLayer + ffnParamsPerLayer + normParamsPerLayer) + ' \u00D7 ' + L + ' = ' + formatNum(totalLayerParams) + '\n';
    pdHtml += '<b>Embedding:</b> h \u00D7 v = ' + h + ' \u00D7 ' + v + ' = ' + formatNum(embedParams) + '\n';
    if (!tiedEmbed) {
      pdHtml += '<b>LM Head:</b> h \u00D7 v = ' + h + ' \u00D7 ' + v + ' = ' + formatNum(lmHeadParams) + ' (untied)\n';
    } else {
      pdHtml += '<b>LM Head:</b> tied with embedding (0 extra params)\n';
    }
    pdHtml += '<b>Final norm:</b> ' + formatNum(finalNormParams) + '\n';
    pdHtml += '\n<b>Total params: ' + formatNum(totalParams) + ' (' + (totalParams / 1e9).toFixed(2) + 'B)</b>\n';
    pdHtml += '<b>Per GPU (TP=' + tp + '): ' + formatNum(totalParamsPerGpu) + '</b>';
    $('kv-param-detail').innerHTML = pdHtml;

    // KV Cache formula breakdown
    var kvHtml = '';
    kvHtml += '<b>KV Cache (per GPU)</b> = batch_size \u00D7 num_layers \u00D7 num_kv_heads \u00D7 head_dim \u00D7 context_length \u00D7 2 \u00D7 dtype_bytes / tp_size\n';
    kvHtml += '= ' + b + ' \u00D7 ' + L + ' \u00D7 ' + k + ' \u00D7 ' + headDim + ' \u00D7 ' + s + ' \u00D7 2 \u00D7 ' + kvBytes + ' / ' + tp + '\n';
    kvHtml += '= <b>' + formatBytes(kvCacheMem) + '</b>';
    $('kv-cache-detail').innerHTML = kvHtml;

    // ===== Stacked Bar Chart =====
    renderBarChart(weightMem, kvCacheMem, actMem, totalMem);

    // ===== GPU Fit Check =====
    renderGpuFit(totalMem, gpuKey);

    // ===== Roofline Analysis =====
    renderRoofline(h, h_ffn, a, k, headDim, s, b, tp, wBytes, actBytes, gpuKey, isMoE, numExperts, topK, expertHffn);
  }

  function renderBarChart(weights, kv, act, total) {
    var container = $('kv-bar-chart');
    if (total <= 0) { container.innerHTML = ''; return; }
    var pW = (weights / total * 100).toFixed(1);
    var pK = (kv / total * 100).toFixed(1);
    var pA = (act / total * 100).toFixed(1);

    var colors = { weights: '#8b5cf6', kv: '#3b82f6', act: '#f59e0b' };

    container.innerHTML =
      '<div class="tool-bar-label">Total: ' + formatBytes(total) + '</div>' +
      '<div class="tool-bar-track">' +
      '<div class="tool-bar-seg" style="flex-basis:' + pW + '%;background:' + colors.weights + '">' + (parseFloat(pW) > 8 ? pW + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pK + '%;background:' + colors.kv + '">' + (parseFloat(pK) > 8 ? pK + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pA + '%;background:' + colors.act + '">' + (parseFloat(pA) > 8 ? pA + '%' : '') + '</div>' +
      '</div>' +
      '<div class="tool-bar-legend">' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.weights + '"></div>Weights ' + formatBytes(weights) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.kv + '"></div>KV Cache ' + formatBytes(kv) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.act + '"></div>Activation ' + formatBytes(act) + '</div>' +
      '</div>';
  }

  function renderGpuFit(totalMem, selectedKey) {
    var container = $('kv-gpu-cards');
    var html = '';
    var keys = Object.keys(GPUS);
    for (var i = 0; i < keys.length; i++) {
      var gk = keys[i];
      var g = GPUS[gk];
      var pct = (totalMem / g.mem * 100).toFixed(1);
      var cls = 'ok';
      var icon = '\u2705';
      if (pct > 100) { cls = 'ng'; icon = '\u274C'; }
      else if (pct > 80) { cls = 'warning'; icon = '\u26A0\uFE0F'; }
      var sel = gk === selectedKey ? ' selected' : '';
      html += '<div class="tool-gpu-card ' + cls + sel + '">' +
        '<div class="tool-gpu-card-icon">' + icon + '</div>' +
        '<div class="tool-gpu-card-name">' + g.name + '</div>' +
        '<div class="tool-gpu-card-pct">' + pct + '%</div>' +
        '</div>';
    }
    container.innerHTML = html;
  }

  function renderRoofline(h, h_ffn, a, k, headDim, s, b, tp, wBytes, actBytes, gpuKey, isMoE, numExperts, topK, expertHffn) {
    var tbody = $('kv-op-tbody');
    var gpu = GPUS[gpuKey];
    var opsPerByte = gpu.bf16Tflops * 1e12 / (gpu.bwTBs * 1e12);

    var rows = [];
    var ffnDim = isMoE ? expertHffn : h_ffn;

    // Decode: 1 new token, KV cache has s tokens
    // QKV Projection: [b, h] x [h, a*headDim + 2*k*headDim]
    var qkvOut = a * headDim + 2 * k * headDim;
    var qkvFlops = 2 * b * h * qkvOut / tp;
    var qkvBytesLoad = h * qkvOut * wBytes / tp + b * h * actBytes;
    var qkvAI = qkvFlops / qkvBytesLoad;
    rows.push({ name: 'QKV Projection', flops: qkvFlops, bytes: qkvBytesLoad, ai: qkvAI, bound: qkvAI < opsPerByte ? 'memory' : 'compute' });

    // Attention Score: Q [b, a/tp, 1, headDim] x K^T [b, k/tp, headDim, s]
    var attnFlops = 2 * b * (a / tp) * s * headDim;
    var attnBytesLoad = b * (k / tp) * s * headDim * actBytes + b * (a / tp) * headDim * actBytes;
    var attnAI = attnFlops / attnBytesLoad;
    rows.push({ name: 'Attn Score (QK\u1D40)', flops: attnFlops, bytes: attnBytesLoad, ai: attnAI, bound: attnAI < opsPerByte ? 'memory' : 'compute' });

    // Attention x V
    var attnVFlops = 2 * b * (a / tp) * headDim * s;
    var attnVBytes = b * (k / tp) * s * headDim * actBytes + b * (a / tp) * s * actBytes;
    var attnVAI = attnVFlops / attnVBytes;
    rows.push({ name: 'Attn \u00D7 V', flops: attnVFlops, bytes: attnVBytes, ai: attnVAI, bound: attnVAI < opsPerByte ? 'memory' : 'compute' });

    // Output Projection: [b, a*headDim] x [a*headDim, h]
    var oDim = a * headDim;
    var oProjFlops = 2 * b * oDim * h / tp;
    var oProjBytes = oDim * h * wBytes / tp + b * oDim * actBytes;
    var oProjAI = oProjFlops / oProjBytes;
    rows.push({ name: 'Output Projection', flops: oProjFlops, bytes: oProjBytes, ai: oProjAI, bound: oProjAI < opsPerByte ? 'memory' : 'compute' });

    // FFN (SwiGLU)
    var ffnMult = isMoE ? topK : 1;
    var ffnFlops = 2 * b * h * ffnDim * 3 * ffnMult / tp;
    var ffnWeightBytes = 3 * h * ffnDim * wBytes * ffnMult / tp;
    var ffnActBytesTotal = b * h * actBytes * 3;
    var ffnTotalBytes = ffnWeightBytes + ffnActBytesTotal;
    var ffnAI = ffnFlops / ffnTotalBytes;
    rows.push({
      name: isMoE ? 'FFN (MoE, top-' + topK + ', dim=' + ffnDim + ')' : 'FFN (SwiGLU)',
      flops: ffnFlops, bytes: ffnTotalBytes, ai: ffnAI,
      bound: ffnAI < opsPerByte ? 'memory' : 'compute'
    });

    // RMSNorm
    var normFlops = 5 * b * h * 2;
    var normBytes = (2 * b * h * actBytes + h * wBytes) * 2;
    var normAI = normFlops / normBytes;
    rows.push({ name: 'RMSNorm (\u00D72)', flops: normFlops, bytes: normBytes, ai: normAI, bound: 'memory' });

    var html = '';
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      html += '<tr>' +
        '<td>' + r.name + '</td>' +
        '<td>' + formatFlops(r.flops) + '</td>' +
        '<td>' + formatBytes(r.bytes) + '</td>' +
        '<td>' + r.ai.toFixed(1) + '</td>' +
        '<td><span class="tool-bound-tag ' + r.bound + '">' + (r.bound === 'memory' ? 'Mem Bound' : 'Compute') + '</span></td>' +
        '</tr>';
    }
    html += '<tr style="border-top:1px solid var(--border);font-size:10px;color:var(--secondary)">' +
      '<td colspan="5">GPU ops:byte ratio (' + gpu.name + '): ' + opsPerByte.toFixed(0) + ' FLOPs/byte (BF16 dense, no sparsity)</td></tr>';
    tbody.innerHTML = html;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
