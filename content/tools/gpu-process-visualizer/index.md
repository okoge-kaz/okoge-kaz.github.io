---
title: "GPU Process Mapping Visualizer"
layout: "single"
toolsJS: gpu-process-visualizer
ShowReadingTime: false
ShowToc: true
---

Visualize how GPU processes are distributed across nodes in distributed LLM training. Configure cluster and parallelism sizes (TP, EP, CP, DP, PP), and see an interactive visualization of process-to-GPU mapping with color-coded communicator groups.

<div class="tool-container" id="gpu-viz-tool">

<div class="tool-form">

<div class="tool-section-label">Cluster Configuration</div>

<div class="tool-input-group">
<label>Number of Nodes</label>
<select id="gpu-num-nodes">
<option value="1">1</option>
<option value="2">2</option>
<option value="4">4</option>
<option value="8">8</option>
<option value="16">16</option>
<option value="32">32</option>
<option value="64">64</option>
</select>
</div>

<div class="tool-input-group">
<label>GPUs per Node</label>
<select id="gpu-gpus-per-node">
<option value="1">1</option>
<option value="2">2</option>
<option value="4">4</option>
<option value="8" selected>8</option>
</select>
</div>

<div class="tool-section-label">Parallelism Configuration</div>

<div class="tool-input-group">
<label>TP Size (Tensor Parallel)</label>
<input type="number" id="gpu-tp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>EP Size (Expert Parallel)</label>
<input type="number" id="gpu-ep" value="1" min="1">
</div>

<div class="tool-input-group">
<label>CP Size (Context Parallel)</label>
<input type="number" id="gpu-cp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>DP Size (Data Parallel)</label>
<input type="number" id="gpu-dp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>PP Size (Pipeline Parallel)</label>
<input type="number" id="gpu-pp" value="1" min="1">
</div>

<div class="tool-input-group">
<label>Preset Configurations</label>
<select id="gpu-preset" class="tool-preset-select">
<option value="custom">Custom</option>
<option value="1n8g-tp8">1N×8G: TP=8</option>
<option value="2n8g-tp8-pp2">2N×8G: TP=8, PP=2</option>
<option value="4n8g-tp8-dp2-pp2">4N×8G: TP=8, DP=2, PP=2</option>
<option value="8n8g-tp8-dp4-pp2">8N×8G: TP=8, DP=4, PP=2</option>
<option value="16n8g-tp8-dp2-pp4-cp2">16N×8G: TP=8, DP=2, PP=4, CP=2</option>
<option value="32n8g-tp8-ep8-dp4-pp2">32N×8G: TP=8, EP=8, DP=4, PP=2 (MoE)</option>
</select>
</div>

</div>

<div class="tool-validation" id="gpu-validation"></div>

<div class="gpu-viz-controls" id="gpu-viz-controls">
<button class="gpu-viz-comm-btn active" data-comm="tp">TP</button>
<button class="gpu-viz-comm-btn" data-comm="ep">EP</button>
<button class="gpu-viz-comm-btn" data-comm="cp">CP</button>
<button class="gpu-viz-comm-btn" data-comm="dp">DP</button>
<button class="gpu-viz-comm-btn" data-comm="pp">PP</button>
<button class="gpu-viz-comm-btn" data-comm="none">None</button>
</div>

<div class="gpu-viz-legend" id="gpu-viz-legend"></div>

<div class="gpu-viz-container" id="gpu-viz-container">
<div class="gpu-viz-scene" id="gpu-viz-scene"></div>
</div>

<div class="gpu-viz-tooltip" id="gpu-viz-tooltip"></div>

</div>

---

## Communicator Basics

In distributed training, GPUs are organized into **process groups** (communicators). Each group defines a set of GPUs that need to communicate for a specific type of parallelism. A single GPU belongs to multiple communicator groups simultaneously — one for each parallelism dimension.

### Rank Mapping Formula

Megatron-LM assigns global ranks using a nested loop where **TP is innermost** (fastest varying) and **PP is outermost** (slowest varying):

```
rank = tp_i + TP × (ep_i + EP × (cp_i + CP × (dp_i + DP × pp_i)))
```

This means GPUs with consecutive ranks share the same TP group, which is critical because TP requires the highest communication bandwidth (NVLink within a node).

### Group Identification

Two GPUs belong to the **same communicator group** if they share the same indices in all *other* dimensions. For example, two GPUs are in the same TP group if they have the same `(ep_i, cp_i, dp_i, pp_i)` but different `tp_i`.

---

## Collective Operations

Each parallelism dimension uses specific NCCL collective operations:

### AllReduce (TP, DP)

<img src="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png" alt="AllReduce" style="max-width: 500px;">

Every rank contributes data and receives the fully reduced result. Used by:
- **TP**: Synchronize partial activation results after column/row-parallel linear layers (every layer, forward + backward)
- **DP**: Gradient synchronization across data-parallel replicas (without ZeRO / distributed optimizer)

### ReduceScatter (DP with ZeRO)

<img src="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png" alt="ReduceScatter" style="max-width: 500px;">

Reduces data and scatters chunks to different ranks. Used by:
- **DP with ZeRO/distributed optimizer**: Each rank receives only its shard of the reduced gradients

### AllGather (DP with ZeRO, TP)

<img src="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png" alt="AllGather" style="max-width: 500px;">

Each rank contributes a chunk; all ranks receive the full concatenated result. Used by:
- **DP with ZeRO**: Gather full parameters before forward/backward passes
- **TP**: Column-parallel output gathering

### AllToAll (EP)

<img src="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/alltoall.png" alt="AllToAll" style="max-width: 500px;">

Each rank sends different data to every other rank (personalized exchange). Used by:
- **EP**: MoE expert routing — dispatch tokens to assigned experts and combine results back

### Send/Recv — Point-to-Point (PP)

Pipeline parallelism uses **P2P Send/Recv** between adjacent pipeline stages. Stage `k` sends activations to stage `k+1` during forward, and stage `k+1` sends gradients back to stage `k` during backward.

---

## Why This Ordering? (TP innermost, PP outermost)

The ordering is determined by **communication bandwidth requirements**:

| Parallelism | Collective | Frequency | Volume per Op | Bandwidth Need |
|---|---|---|---|---|
| **TP** | AllReduce | Every layer (fwd+bwd) | O(b × s × h) | **Highest** |
| **EP** | AllToAll | Every MoE layer | O(b × s × h × top_k / E) | High |
| **CP** | P2P Ring + AllGather | Every attention layer | O(b × s × h / CP) | Moderate |
| **DP** | AllReduce / ReduceScatter | Once per micro-step | O(params), overlapped | Low (overlapped) |
| **PP** | P2P Send/Recv | Per micro-batch boundary | O(b × s × h) per boundary | **Lowest** |

### Key Insight

- **TP communicates EVERY layer** → needs the highest total bandwidth → must use NVLink (intra-node, 900 GB/s on H100)
- **EP communicates every MoE layer** → high bandwidth → typically intra-node or single-hop inter-node
- **DP communicates ONCE per step** + overlaps with backward compute → tolerates higher latency → can span nodes
- **PP only sends activations between adjacent stages** → minimal total volume, latency-tolerant → outermost, can span multiple network hops

By placing TP innermost, consecutive rank IDs map to GPUs on the **same node** connected by NVLink. PP outermost means pipeline stages span **across nodes**, which is fine because PP communication is infrequent and small.
