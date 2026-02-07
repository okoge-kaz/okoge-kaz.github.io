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

  // h_ffn = intermediate_size (dense), expertHffn = moe_intermediate_size (per-expert)
  var PRESETS = {
    'llama3-8b':      { h: 4096,  h_ffn: 14336, L: 32,  a: 32,  k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 14336, topk: 1, seqLen: 4096  },
    'llama3-70b':     { h: 8192,  h_ffn: 28672, L: 80,  a: 64,  k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 28672, topk: 1, seqLen: 4096  },
    'llama3-405b':    { h: 16384, h_ffn: 53248, L: 126, a: 128, k: 8,   headDim: 128, v: 128256, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 53248, topk: 1, seqLen: 4096  },
    'qwen3-8b':       { h: 4096,  h_ffn: 12288, L: 36,  a: 32,  k: 8,   headDim: 128, v: 151936, tiedEmbed: true,  experts: 1,   sharedExperts: 0, expertHffn: 12288, topk: 1, seqLen: 4096  },
    'qwen3-72b':      { h: 8192,  h_ffn: 29568, L: 80,  a: 64,  k: 8,   headDim: 128, v: 152064, tiedEmbed: false, experts: 1,   sharedExperts: 0, expertHffn: 29568, topk: 1, seqLen: 4096  },
    'qwen3-235b-moe': { h: 4096,  h_ffn: 12288, L: 94,  a: 64,  k: 4,   headDim: 64,  v: 151936, tiedEmbed: true,  experts: 128, sharedExperts: 0, expertHffn: 1536,  topk: 8, seqLen: 4096  },
    'deepseek-v3':    { h: 7168,  h_ffn: 18432, L: 61,  a: 128, k: 128, headDim: 128, v: 129280, tiedEmbed: false, experts: 256, sharedExperts: 1, expertHffn: 2048,  topk: 8, seqLen: 4096  },
  };

  var els = {};

  function $(id) { return document.getElementById(id); }

  function init() {
    els.preset       = $('mem-preset');
    els.hiddenSize   = $('mem-hidden-size');
    els.interSize    = $('mem-intermediate-size');
    els.layers       = $('mem-num-layers');
    els.numHeads     = $('mem-num-heads');
    els.kvHeads      = $('mem-num-kv-heads');
    els.headDim      = $('mem-head-dim');
    els.vocabSize    = $('mem-vocab-size');
    els.numExperts   = $('mem-num-experts');
    els.sharedExperts= $('mem-shared-experts');
    els.expertFfn    = $('mem-expert-ffn');
    els.topK         = $('mem-experts-per-token');
    els.tieEmbed     = $('mem-tie-embed');
    els.paramDtype   = $('mem-param-dtype');
    els.gradDtype    = $('mem-grad-dtype');
    els.optimizer    = $('mem-optimizer');
    els.seqLen       = $('mem-seq-len');
    els.microBatch   = $('mem-micro-batch');
    els.tp           = $('mem-tp');
    els.pp           = $('mem-pp');
    els.dp           = $('mem-dp');
    els.cp           = $('mem-cp');
    els.ep           = $('mem-ep');
    els.distOptim    = $('mem-dist-optim');
    els.ppSched      = $('mem-pp-sched');
    els.vpp          = $('mem-vpp');
    els.gpuType      = $('mem-gpu-type');

    els.preset.addEventListener('change', applyPreset);

    var inputs = document.querySelectorAll('#mem-tool input, #mem-tool select');
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
    els.hiddenSize.value    = p.h;
    els.interSize.value     = p.h_ffn;
    els.layers.value        = p.L;
    els.numHeads.value      = p.a;
    els.kvHeads.value       = p.k;
    els.headDim.value       = p.headDim;
    els.vocabSize.value     = p.v;
    els.numExperts.value    = p.experts;
    els.sharedExperts.value = p.sharedExperts;
    els.expertFfn.value     = p.expertHffn;
    els.topK.value          = p.topk;
    els.tieEmbed.value      = p.tiedEmbed ? '1' : '0';
    els.seqLen.value        = p.seqLen;
    calculate();
  }

  // ===== HuggingFace Config Fetch =====
  window.memFetchHF = function () {
    var modelId = $('mem-hf-model-id').value.trim();
    var statusEl = $('mem-hf-status');
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
        // Head dim
        if (cfg.head_dim) {
          els.headDim.value = cfg.head_dim;
        } else if (cfg.v_head_dim) {
          els.headDim.value = cfg.v_head_dim;
        } else if (cfg.hidden_size && cfg.num_attention_heads) {
          els.headDim.value = Math.floor(cfg.hidden_size / cfg.num_attention_heads);
        }
        if (cfg.vocab_size) els.vocabSize.value = cfg.vocab_size;

        // MoE fields
        var nExperts = cfg.n_routed_experts || cfg.num_local_experts || cfg.num_experts || 1;
        els.numExperts.value = nExperts;
        els.sharedExperts.value = cfg.n_shared_experts || 0;

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
          els.topK.value = 1;
        }

        // tie_word_embeddings: default is true in HF if not specified
        if (cfg.tie_word_embeddings === false) {
          els.tieEmbed.value = '0';
        } else {
          els.tieEmbed.value = '1';
        }

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
    var paramBytes = parseFloat(els.paramDtype.value) || 2;
    var gradBytes  = parseFloat(els.gradDtype.value) || 2;
    var optimType  = els.optimizer.value;
    var s     = parseInt(els.seqLen.value) || 0;
    var b     = parseInt(els.microBatch.value) || 1;
    var tp    = parseInt(els.tp.value) || 1;
    var pp    = parseInt(els.pp.value) || 1;
    var dp    = parseInt(els.dp.value) || 1;
    var cp    = parseInt(els.cp.value) || 1;
    var ep    = parseInt(els.ep.value) || 1;
    var distOptim = parseInt(els.distOptim.value);
    var ppSched   = els.ppSched.value;
    var vpp       = parseInt(els.vpp.value) || 1;
    var gpuKey    = els.gpuType.value;

    var isMoE = numExperts > 1;
    var optimBytesPerParam = optimType === 'adam' ? 12 : 8;

    // ===== Architecture Info =====
    var archInfo = getAttnType(a, k);
    if (isMoE) archInfo += ' | MoE (' + numExperts + ' routed + ' + sharedExperts + ' shared, top-' + topK + ', expert_ffn=' + expertHffn + ')';
    else archInfo += ' | Dense FFN';
    archInfo += ' | head_dim=' + headDim;
    $('mem-arch-info').textContent = archInfo;

    // ===== Parameter Count =====
    // Attention: Q(h, a*headDim) + K(h, k*headDim) + V(h, k*headDim) + O(a*headDim, h)
    var attnParamsPerLayer = 2 * h * a * headDim + 2 * h * k * headDim;

    // FFN per layer
    var ffnParamsPerLayer;
    if (isMoE) {
      ffnParamsPerLayer = numExperts * 3 * h * expertHffn
                        + sharedExperts * 3 * h * expertHffn
                        + h * numExperts; // router gate
    } else {
      ffnParamsPerLayer = 3 * h * h_ffn;
    }

    // RMSNorm: 2 per layer
    var normParamsPerLayer = 2 * h;

    var totalLayerParams = (attnParamsPerLayer + ffnParamsPerLayer + normParamsPerLayer) * L;
    var embedParams = h * v;
    var lmHeadParams = tiedEmbed ? 0 : h * v;
    var finalNormParams = h;
    var totalParams = totalLayerParams + embedParams + lmHeadParams + finalNormParams;

    // ===== Per-GPU Parameters =====
    var layersPerStage = Math.ceil(L / pp);

    var attnParamsPerGpu = (attnParamsPerLayer * layersPerStage) / tp;
    var ffnParamsPerGpu;
    if (isMoE) {
      ffnParamsPerGpu = (ffnParamsPerLayer * layersPerStage) / (tp * ep);
    } else {
      ffnParamsPerGpu = (ffnParamsPerLayer * layersPerStage) / tp;
    }
    var normParamsPerGpu = normParamsPerLayer * layersPerStage;
    var embedParamsPerGpu = (embedParams + lmHeadParams) / tp + finalNormParams;

    var totalParamsPerGpu = attnParamsPerGpu + ffnParamsPerGpu + normParamsPerGpu + embedParamsPerGpu;

    // ===== Model States Memory =====
    var paramsMem = totalParamsPerGpu * paramBytes;
    var gradsMem = totalParamsPerGpu * gradBytes;

    var optimMem;
    if (distOptim) {
      optimMem = totalParamsPerGpu * optimBytesPerParam / dp;
    } else {
      optimMem = totalParamsPerGpu * optimBytesPerParam;
    }

    // ===== Activation Memory =====
    var actBytes = 2; // BF16

    // Per layer activations (FlashAttention)
    var attnActPerLayer = s * b * h * (6 + 4 * k / a) * actBytes;

    var ffnActPerLayer;
    if (isMoE) {
      ffnActPerLayer = 2 * s * b * (h + 4 * expertHffn) * topK * actBytes;
    } else {
      ffnActPerLayer = 2 * s * b * (h + 4 * h_ffn) * actBytes;
    }

    var normActPerLayer = 4 * s * b * h * actBytes;
    var actPerLayer = attnActPerLayer + ffnActPerLayer + normActPerLayer;
    var actPerLayerPerGpu = actPerLayer / (tp * cp);

    // Pipeline inflight microbatches
    var numInflight;
    if (pp === 1) {
      numInflight = 1;
    } else {
      numInflight = pp;
    }

    var totalAct;
    if (pp === 1) {
      totalAct = actPerLayerPerGpu * L;
    } else {
      var effectiveLayersPerStage = layersPerStage;
      if (ppSched === 'interleaved' && vpp > 1) {
        effectiveLayersPerStage = Math.ceil(L / (pp * vpp));
      }
      totalAct = actPerLayerPerGpu * effectiveLayersPerStage * numInflight;
    }

    // ===== NCCL Buffer Estimation =====
    var numComms = 0;
    if (tp > 1) numComms++;
    if (pp > 1) numComms++;
    if (dp > 1) numComms++;
    if (cp > 1) numComms++;
    if (ep > 1) numComms++;
    var ncclMem = numComms * 0.5e9;
    if (numComms === 0) ncclMem = 0;

    // ===== Total =====
    var totalMem = paramsMem + gradsMem + optimMem + totalAct + ncclMem;

    // ===== Update Results =====
    $('mem-r-params').textContent = formatBytes(paramsMem);
    $('mem-f-params').textContent = formatNum(totalParamsPerGpu) + ' \u00D7 ' + paramBytes + 'B = ' + formatBytes(paramsMem);

    $('mem-r-grad').textContent = formatBytes(gradsMem);
    $('mem-f-grad').textContent = formatNum(totalParamsPerGpu) + ' \u00D7 ' + gradBytes + 'B = ' + formatBytes(gradsMem);

    $('mem-r-optim').textContent = formatBytes(optimMem);
    var optimFormula = formatNum(totalParamsPerGpu) + ' \u00D7 ' + optimBytesPerParam + 'B';
    if (distOptim) optimFormula += ' / dp(' + dp + ')';
    $('mem-f-optim').textContent = optimFormula + ' = ' + formatBytes(optimMem);

    $('mem-r-act').textContent = formatBytes(totalAct);
    var actFormula = 'act/layer=' + formatBytes(actPerLayerPerGpu);
    if (pp === 1) {
      actFormula += ' \u00D7 ' + L + ' layers';
    } else {
      var effLayers = (ppSched === 'interleaved' && vpp > 1) ? Math.ceil(L / (pp * vpp)) : layersPerStage;
      actFormula += ' \u00D7 ' + effLayers + ' layers \u00D7 ' + numInflight + ' inflight';
    }
    $('mem-f-act').textContent = actFormula;

    $('mem-r-nccl').textContent = formatBytes(ncclMem);
    $('mem-f-nccl').textContent = numComms + ' communicators \u00D7 ~0.5 GB';

    $('mem-r-total').textContent = formatBytes(totalMem);

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
    pdHtml += '<b>Per GPU (TP=' + tp + ', PP=' + pp + '): ' + formatNum(totalParamsPerGpu) + '</b>';
    if (isMoE && ep > 1) pdHtml += ' (EP=' + ep + ' applied to FFN)';
    pdHtml += '\n<b>Layers/stage:</b> ' + layersPerStage;
    $('mem-param-detail').innerHTML = pdHtml;

    renderBarChart(paramsMem, gradsMem, optimMem, totalAct, ncclMem, totalMem);
    renderGpuFit(totalMem, gpuKey);
    renderRoofline(h, h_ffn, a, k, headDim, s, b, tp, actBytes, gpuKey, isMoE, numExperts, topK, expertHffn);
  }

  function renderBarChart(params, grads, optim, act, nccl, total) {
    var container = $('mem-bar-chart');
    if (total <= 0) { container.innerHTML = ''; return; }
    var pP = (params / total * 100).toFixed(1);
    var pG = (grads / total * 100).toFixed(1);
    var pO = (optim / total * 100).toFixed(1);
    var pA = (act / total * 100).toFixed(1);
    var pN = (nccl / total * 100).toFixed(1);

    var colors = { params: '#8b5cf6', grads: '#06b6d4', optim: '#f59e0b', act: '#ef4444', nccl: '#6b7280' };

    container.innerHTML =
      '<div class="tool-bar-label">Total: ' + formatBytes(total) + '</div>' +
      '<div class="tool-bar-track">' +
      '<div class="tool-bar-seg" style="flex-basis:' + pP + '%;background:' + colors.params + '">' + (parseFloat(pP) > 6 ? pP + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pG + '%;background:' + colors.grads + '">' + (parseFloat(pG) > 6 ? pG + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pO + '%;background:' + colors.optim + '">' + (parseFloat(pO) > 6 ? pO + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pA + '%;background:' + colors.act + '">' + (parseFloat(pA) > 6 ? pA + '%' : '') + '</div>' +
      '<div class="tool-bar-seg" style="flex-basis:' + pN + '%;background:' + colors.nccl + '">' + (parseFloat(pN) > 6 ? pN + '%' : '') + '</div>' +
      '</div>' +
      '<div class="tool-bar-legend">' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.params + '"></div>Params ' + formatBytes(params) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.grads + '"></div>Grads ' + formatBytes(grads) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.optim + '"></div>Optim ' + formatBytes(optim) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.act + '"></div>Activations ' + formatBytes(act) + '</div>' +
      '<div class="tool-bar-legend-item"><div class="tool-bar-legend-swatch" style="background:' + colors.nccl + '"></div>NCCL ' + formatBytes(nccl) + '</div>' +
      '</div>';
  }

  function renderGpuFit(totalMem, selectedKey) {
    var container = $('mem-gpu-cards');
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

  function renderRoofline(h, h_ffn, a, k, headDim, s, b, tp, actBytes, gpuKey, isMoE, numExperts, topK, expertHffn) {
    var tbody = $('mem-op-tbody');
    var gpu = GPUS[gpuKey];
    var opsPerByte = gpu.bf16Tflops * 1e12 / (gpu.bwTBs * 1e12);
    var ffnDim = isMoE ? expertHffn : h_ffn;

    var rows = [];
    var tokens = s * b;

    // QKV Projection (fwd + 2x bwd)
    var qkvOut = a * headDim + 2 * k * headDim;
    var qkvFlops = 2 * tokens * h * qkvOut / tp * 3;
    var qkvWeightBytes = h * qkvOut * actBytes / tp;
    var qkvActBytesTotal = tokens * (h + qkvOut) * actBytes;
    var qkvTotalBytes = qkvWeightBytes + qkvActBytesTotal;
    var qkvAI = qkvFlops / qkvTotalBytes;
    rows.push({ name: 'QKV Projection', flops: qkvFlops, bytes: qkvTotalBytes, ai: qkvAI, bound: qkvAI < opsPerByte ? 'memory' : 'compute' });

    // Self-Attention (FlashAttention)
    var attnFlops = 4 * tokens * s * a * headDim / tp * 2;
    var attnBytesTotal = tokens * h * actBytes * 4 / tp;
    var attnAI = attnFlops / attnBytesTotal;
    rows.push({ name: 'Self-Attention (FA)', flops: attnFlops, bytes: attnBytesTotal, ai: attnAI, bound: attnAI < opsPerByte ? 'memory' : 'compute' });

    // Output Projection
    var oDim = a * headDim;
    var oProjFlops = 2 * tokens * oDim * h / tp * 3;
    var oProjBytes = oDim * h * actBytes / tp + tokens * (oDim + h) * actBytes;
    var oProjAI = oProjFlops / oProjBytes;
    rows.push({ name: 'Output Projection', flops: oProjFlops, bytes: oProjBytes, ai: oProjAI, bound: oProjAI < opsPerByte ? 'memory' : 'compute' });

    // FFN (SwiGLU)
    var ffnMult = isMoE ? topK : 1;
    var ffnFlops = 2 * tokens * h * ffnDim * 3 * ffnMult / tp * 3;
    var ffnWeightBytes = 3 * h * ffnDim * actBytes * ffnMult / tp;
    var ffnActBytesTotal = tokens * (h + ffnDim * 3) * actBytes * ffnMult;
    var ffnTotalBytes = ffnWeightBytes + ffnActBytesTotal;
    var ffnAI = ffnFlops / ffnTotalBytes;
    rows.push({
      name: isMoE ? 'FFN (MoE, top-' + topK + ', dim=' + ffnDim + ')' : 'FFN (SwiGLU)',
      flops: ffnFlops, bytes: ffnTotalBytes, ai: ffnAI,
      bound: ffnAI < opsPerByte ? 'memory' : 'compute'
    });

    // RMSNorm
    var normFlops = 5 * tokens * h * 2 * 3;
    var normBytes = tokens * h * actBytes * 4 * 2 * 3;
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
      '<td colspan="5">GPU ops:byte ratio (' + gpu.name + '): ' + opsPerByte.toFixed(0) + ' FLOPs/byte (BF16 dense) | Training = fwd + 2\u00D7bwd per op</td></tr>';
    tbody.innerHTML = html;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
