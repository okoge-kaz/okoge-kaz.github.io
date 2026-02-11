(function() {
  'use strict';

  var COMMS = {
    tp: { name: 'Tensor Parallel (TP)', hue: 271 },
    ep: { name: 'Expert Parallel (EP)', hue: 30 },
    cp: { name: 'Context Parallel (CP)', hue: 140 },
    dp: { name: 'Data Parallel (DP)', hue: 211 },
    pp: { name: 'Pipeline Parallel (PP)', hue: 0 }
  };

  var PRESETS = {
    '1n8g-tp8':              { nodes: 1,  gpus: 8, tp: 8, ep: 1, cp: 1, dp: 1, pp: 1 },
    '2n8g-tp8-pp2':          { nodes: 2,  gpus: 8, tp: 8, ep: 1, cp: 1, dp: 1, pp: 2 },
    '4n8g-tp8-dp2-pp2':      { nodes: 4,  gpus: 8, tp: 8, ep: 1, cp: 1, dp: 2, pp: 2 },
    '8n8g-tp8-dp4-pp2':      { nodes: 8,  gpus: 8, tp: 8, ep: 1, cp: 1, dp: 4, pp: 2 },
    '16n8g-tp8-dp2-pp4-cp2': { nodes: 16, gpus: 8, tp: 8, ep: 1, cp: 2, dp: 2, pp: 4 },
    '32n8g-tp8-ep8-dp4-pp2': { nodes: 32, gpus: 8, tp: 8, ep: 8, cp: 1, dp: 4, pp: 2 }
  };

  var state = {
    activeComm: 'tp',
    highlightGroup: -1,
    ranks: [],
    dragStartX: 0,
    dragStartY: 0,
    rotX: -8,
    rotY: 12,
    dragging: false
  };

  var dom = {};

  function init() {
    dom.numNodes = document.getElementById('gpu-num-nodes');
    dom.gpusPerNode = document.getElementById('gpu-gpus-per-node');
    dom.tp = document.getElementById('gpu-tp');
    dom.ep = document.getElementById('gpu-ep');
    dom.cp = document.getElementById('gpu-cp');
    dom.dp = document.getElementById('gpu-dp');
    dom.pp = document.getElementById('gpu-pp');
    dom.preset = document.getElementById('gpu-preset');
    dom.validation = document.getElementById('gpu-validation');
    dom.controls = document.getElementById('gpu-viz-controls');
    dom.legend = document.getElementById('gpu-viz-legend');
    dom.container = document.getElementById('gpu-viz-container');
    dom.scene = document.getElementById('gpu-viz-scene');
    dom.tooltip = document.getElementById('gpu-viz-tooltip');

    if (!dom.scene) return;

    // Bind form changes
    var inputs = [dom.numNodes, dom.gpusPerNode, dom.tp, dom.ep, dom.cp, dom.dp, dom.pp];
    for (var i = 0; i < inputs.length; i++) {
      inputs[i].addEventListener('change', onFormChange);
      inputs[i].addEventListener('input', onFormChange);
    }

    dom.preset.addEventListener('change', onPresetChange);

    // Bind communicator toggle buttons
    var btns = dom.controls.querySelectorAll('.gpu-viz-comm-btn');
    for (var b = 0; b < btns.length; b++) {
      btns[b].addEventListener('click', onCommToggle);
    }

    // 3D drag
    dom.container.addEventListener('mousedown', onDragStart);
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);

    // Apply default preset
    applyPreset('1n8g-tp8');
    dom.preset.value = '1n8g-tp8';
  }

  function onFormChange() {
    dom.preset.value = 'custom';
    update();
  }

  function onPresetChange() {
    var key = dom.preset.value;
    if (key !== 'custom' && PRESETS[key]) {
      applyPreset(key);
    }
  }

  function applyPreset(key) {
    var p = PRESETS[key];
    if (!p) return;
    dom.numNodes.value = p.nodes;
    dom.gpusPerNode.value = p.gpus;
    dom.tp.value = p.tp;
    dom.ep.value = p.ep;
    dom.cp.value = p.cp;
    dom.dp.value = p.dp;
    dom.pp.value = p.pp;
    update();
  }

  function onCommToggle(e) {
    var comm = e.target.getAttribute('data-comm');
    state.activeComm = comm;
    state.highlightGroup = -1;

    var btns = dom.controls.querySelectorAll('.gpu-viz-comm-btn');
    for (var i = 0; i < btns.length; i++) {
      btns[i].classList.remove('active');
    }
    e.target.classList.add('active');

    colorize();
    buildLegend();
  }

  function getVal(el) { return parseInt(el.value, 10) || 1; }

  function validate() {
    var nodes = getVal(dom.numNodes);
    var gpus = getVal(dom.gpusPerNode);
    var tp = getVal(dom.tp);
    var ep = getVal(dom.ep);
    var cp = getVal(dom.cp);
    var dp = getVal(dom.dp);
    var pp = getVal(dom.pp);
    var totalGPUs = nodes * gpus;
    var product = tp * ep * cp * dp * pp;
    var errors = [];

    if (product !== totalGPUs) {
      errors.push('TP \u00d7 EP \u00d7 CP \u00d7 DP \u00d7 PP = ' + product +
        ' but total GPUs = ' + nodes + ' \u00d7 ' + gpus + ' = ' + totalGPUs);
    }

    dom.validation.innerHTML = '';
    for (var i = 0; i < errors.length; i++) {
      var div = document.createElement('div');
      div.className = 'tool-validation-item';
      div.textContent = errors[i];
      dom.validation.appendChild(div);
    }

    return errors.length === 0;
  }

  function computeRanks(tp, ep, cp, dp, pp, gpusPerNode) {
    var ranks = [];
    for (var pp_i = 0; pp_i < pp; pp_i++) {
      for (var dp_i = 0; dp_i < dp; dp_i++) {
        for (var cp_i = 0; cp_i < cp; cp_i++) {
          for (var ep_i = 0; ep_i < ep; ep_i++) {
            for (var tp_i = 0; tp_i < tp; tp_i++) {
              var rank = tp_i + tp * (ep_i + ep * (cp_i + cp * (dp_i + dp * pp_i)));
              ranks.push({
                rank: rank,
                tp: tp_i, ep: ep_i, cp: cp_i, dp: dp_i, pp: pp_i,
                node: Math.floor(rank / gpusPerNode),
                gpu: rank % gpusPerNode
              });
            }
          }
        }
      }
    }
    return ranks;
  }

  function getGroupId(r, comm) {
    // Group ID = combination of all OTHER dimension indices
    switch (comm) {
      case 'tp': return r.ep + ',' + r.cp + ',' + r.dp + ',' + r.pp;
      case 'ep': return r.tp + ',' + r.cp + ',' + r.dp + ',' + r.pp;
      case 'cp': return r.tp + ',' + r.ep + ',' + r.dp + ',' + r.pp;
      case 'dp': return r.tp + ',' + r.ep + ',' + r.cp + ',' + r.pp;
      case 'pp': return r.tp + ',' + r.ep + ',' + r.cp + ',' + r.dp;
      default:   return 'none';
    }
  }

  function getUniqueGroups(ranks, comm) {
    if (comm === 'none') return [];
    var seen = {};
    var groups = [];
    for (var i = 0; i < ranks.length; i++) {
      var gid = getGroupId(ranks[i], comm);
      if (!seen[gid]) {
        seen[gid] = true;
        groups.push(gid);
      }
    }
    return groups;
  }

  function generateColors(hue, count) {
    if (count <= 1) return ['hsl(' + hue + ', 70%, 50%)'];
    var colors = [];
    for (var i = 0; i < count; i++) {
      var lightness = 35 + (i / (count - 1)) * 35;
      var saturation = 80 - (i / (count - 1)) * 20;
      colors.push('hsl(' + hue + ', ' + saturation + '%, ' + lightness + '%)');
    }
    return colors;
  }

  function update() {
    if (!validate()) {
      dom.scene.innerHTML = '';
      dom.legend.innerHTML = '';
      return;
    }

    var tp = getVal(dom.tp);
    var ep = getVal(dom.ep);
    var cp = getVal(dom.cp);
    var dp = getVal(dom.dp);
    var pp = getVal(dom.pp);
    var gpusPerNode = getVal(dom.gpusPerNode);

    state.ranks = computeRanks(tp, ep, cp, dp, pp, gpusPerNode);
    state.highlightGroup = -1;

    buildViz();
    colorize();
    buildLegend();
  }

  function buildViz() {
    var gpusPerNode = getVal(dom.gpusPerNode);
    var numNodes = getVal(dom.numNodes);

    dom.scene.innerHTML = '';
    dom.scene.style.transform = 'rotateX(' + state.rotX + 'deg) rotateY(' + state.rotY + 'deg)';

    for (var n = 0; n < numNodes; n++) {
      var nodeEl = document.createElement('div');
      nodeEl.className = 'gpu-node';

      var label = document.createElement('div');
      label.className = 'gpu-node-label';
      label.textContent = 'Node ' + n;
      nodeEl.appendChild(label);

      var grid = document.createElement('div');
      grid.className = 'gpu-node-grid';

      for (var g = 0; g < gpusPerNode; g++) {
        var globalRank = n * gpusPerNode + g;
        var rankInfo = null;
        for (var r = 0; r < state.ranks.length; r++) {
          if (state.ranks[r].rank === globalRank) {
            rankInfo = state.ranks[r];
            break;
          }
        }

        var cell = document.createElement('div');
        cell.className = 'gpu-cell';
        cell.setAttribute('data-rank', globalRank);
        cell.textContent = globalRank;

        if (rankInfo) {
          cell.setAttribute('data-tp', rankInfo.tp);
          cell.setAttribute('data-ep', rankInfo.ep);
          cell.setAttribute('data-cp', rankInfo.cp);
          cell.setAttribute('data-dp', rankInfo.dp);
          cell.setAttribute('data-pp', rankInfo.pp);
          cell.addEventListener('mouseenter', onCellHover);
          cell.addEventListener('mouseleave', onCellLeave);
        }

        grid.appendChild(cell);
      }

      nodeEl.appendChild(grid);
      dom.scene.appendChild(nodeEl);
    }
  }

  function colorize() {
    var comm = state.activeComm;
    var cells = dom.scene.querySelectorAll('.gpu-cell');

    if (comm === 'none') {
      for (var i = 0; i < cells.length; i++) {
        cells[i].style.backgroundColor = '';
        cells[i].style.color = '';
        cells[i].classList.remove('dim', 'highlight');
      }
      return;
    }

    var groups = getUniqueGroups(state.ranks, comm);
    var hue = COMMS[comm].hue;
    var colors = generateColors(hue, groups.length);
    var groupIndex = {};
    for (var g = 0; g < groups.length; g++) {
      groupIndex[groups[g]] = g;
    }

    for (var c = 0; c < cells.length; c++) {
      var rank = parseInt(cells[c].getAttribute('data-rank'), 10);
      var rankInfo = null;
      for (var r = 0; r < state.ranks.length; r++) {
        if (state.ranks[r].rank === rank) {
          rankInfo = state.ranks[r];
          break;
        }
      }
      if (!rankInfo) continue;

      var gid = getGroupId(rankInfo, comm);
      var idx = groupIndex[gid];
      cells[c].style.backgroundColor = colors[idx];
      cells[c].style.color = '#fff';

      if (state.highlightGroup >= 0) {
        if (idx === state.highlightGroup) {
          cells[c].classList.add('highlight');
          cells[c].classList.remove('dim');
        } else {
          cells[c].classList.add('dim');
          cells[c].classList.remove('highlight');
        }
      } else {
        cells[c].classList.remove('dim', 'highlight');
      }
    }
  }

  function buildLegend() {
    dom.legend.innerHTML = '';
    var comm = state.activeComm;
    if (comm === 'none') return;

    var groups = getUniqueGroups(state.ranks, comm);
    var hue = COMMS[comm].hue;
    var colors = generateColors(hue, groups.length);

    var title = document.createElement('span');
    title.className = 'gpu-viz-legend-title';
    title.textContent = COMMS[comm].name + ' Groups (' + groups.length + '):';
    dom.legend.appendChild(title);

    var wrapper = document.createElement('div');
    wrapper.className = 'gpu-viz-legend-items';

    for (var i = 0; i < groups.length; i++) {
      var item = document.createElement('span');
      item.className = 'gpu-viz-legend-item';
      item.setAttribute('data-group-idx', i);

      var swatch = document.createElement('span');
      swatch.className = 'gpu-viz-legend-swatch';
      swatch.style.backgroundColor = colors[i];

      var label = document.createElement('span');
      label.textContent = 'Group ' + i;

      item.appendChild(swatch);
      item.appendChild(label);
      item.addEventListener('click', onLegendClick);
      wrapper.appendChild(item);
    }

    dom.legend.appendChild(wrapper);
  }

  function onLegendClick(e) {
    var item = e.currentTarget;
    var idx = parseInt(item.getAttribute('data-group-idx'), 10);
    if (state.highlightGroup === idx) {
      state.highlightGroup = -1;
    } else {
      state.highlightGroup = idx;
    }
    colorize();
  }

  function onCellHover(e) {
    var cell = e.currentTarget;
    var rank = cell.getAttribute('data-rank');
    var tp = cell.getAttribute('data-tp');
    var ep = cell.getAttribute('data-ep');
    var cp = cell.getAttribute('data-cp');
    var dp = cell.getAttribute('data-dp');
    var pp = cell.getAttribute('data-pp');

    dom.tooltip.innerHTML =
      '<strong>Rank ' + rank + '</strong><br>' +
      'TP=' + tp + ' EP=' + ep + ' CP=' + cp + '<br>' +
      'DP=' + dp + ' PP=' + pp;
    dom.tooltip.style.display = 'block';

    var rect = cell.getBoundingClientRect();
    var containerRect = dom.container.getBoundingClientRect();
    dom.tooltip.style.left = (rect.left - containerRect.left + rect.width / 2) + 'px';
    dom.tooltip.style.top = (rect.top - containerRect.top - 8) + 'px';
  }

  function onCellLeave() {
    dom.tooltip.style.display = 'none';
  }

  function onDragStart(e) {
    if (e.target.classList.contains('gpu-cell')) return;
    state.dragging = true;
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
    dom.container.style.cursor = 'grabbing';
  }

  function onDragMove(e) {
    if (!state.dragging) return;
    var dx = e.clientX - state.dragStartX;
    var dy = e.clientY - state.dragStartY;
    state.rotY = Math.max(-45, Math.min(45, state.rotY + dx * 0.3));
    state.rotX = Math.max(-30, Math.min(10, state.rotX - dy * 0.3));
    dom.scene.style.transform = 'rotateX(' + state.rotX + 'deg) rotateY(' + state.rotY + 'deg)';
    state.dragStartX = e.clientX;
    state.dragStartY = e.clientY;
  }

  function onDragEnd() {
    state.dragging = false;
    dom.container.style.cursor = 'grab';
  }

  document.addEventListener('DOMContentLoaded', init);
})();
