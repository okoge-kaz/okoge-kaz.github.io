---
title: "Training gpt-oss with NVIDIA NeMo"
date: 2025-11-04
tags: ["LLM", "NeMo", "Megatron-LM", "gpt-oss", "HPC", "TransformerEngine"]
description: "A detailed guide on continual pre-training of OpenAI's gpt-oss using NVIDIA NeMo, covering all the technical hurdles and their solutions."
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: false
---

> This post is an English version of my Japanese article on Zenn: [NVIDIA NeMoを利用したGPT-OSSの学習](https://zenn.dev/turing_motors/articles/81cf3128b22c63)

## Introduction

I'm [Kazuki Fujii](https://x.com/okoge_kaz) from the Institute of Science Tokyo.
This article explains how to train **gpt-oss**, released by OpenAI in August 2025, using the **NVIDIA** [NeMo](https://github.com/NVIDIA-NeMo/NeMo) framework.

As of November 4, 2025, the [official NVIDIA documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html) only covers LoRA finetuning. If you want to do serious training such as long-context continual pre-training, there are **many hurdles** to overcome.
This article documents **detailed solutions** for every problem you need to solve. I hope it helps anyone working on model training with gpt-oss.

## gpt-oss

### About

gpt-oss is an LLM released by OpenAI, available in two sizes: [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) and [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b). Both models demonstrate strong language capabilities in English, as shown below.

![Artificial Analysis benchmark](https://storage.googleapis.com/zenn-user-upload/e929fc4f1b11-20251104.png)
*Source: [Artificial Analysis](https://artificialanalysis.ai/models/open-source)*

However, their Japanese knowledge and language ability are limited, leaving room for improvement.

> **Q: Does this mean continual pre-training could be effective for improving Japanese capabilities?**
>
> Yes. However, enhancing Japanese ability without degrading gpt-oss's strong English, math, code, and reasoning capabilities is **far from trivial**. Naively performing continual pre-training or SFT with Japanese data is likely to significantly harm the model's existing strengths, so careful engineering is required.

### Model Architecture

The gpt-oss architecture has several notable differences from recent open LLMs, which raise the barrier for training:

1. **Bias terms**: Since Llama-2, most open LLMs have omitted bias terms in MLP and Attention layers. gpt-oss brings them back, similar to the GPT-2 era.
2. **No QK Norm**: While recent LLMs like Qwen3 have adopted QK Norm for training stability, gpt-oss does not include it.
3. **Self-attention sink (learnable softmax)**: A learnable bias term is introduced in the denominator of the softmax function.

While these architectural changes are unlikely to have a major impact on model performance, the third point in particular causes significant problems during training.

> **Q: Where can I find details about the attention sink?**
>
> The [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/pdf/2508.10925) states: "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks."
>
> You can also confirm `self_attn.sinks` in the [model structure on HuggingFace](https://huggingface.co/openai/gpt-oss-20b/tree/main?show_file_info=model.safetensors.index.json).
>
> ![self_attn.sinks](https://storage.googleapis.com/zenn-user-upload/43431baacaf4-20251104.png)
>
> [Gro Kobayashi's tweet](https://x.com/goro_koba/status/1954480023890780587?s=20) is also a helpful reference.
>
> ![Tweet image](https://storage.googleapis.com/zenn-user-upload/2f6d51dcc228-20251112.png)

> **Q: Why does the learnable softmax cause problems?**
>
> Modern LLM training doesn't run on simple PyTorch implementations alone. It relies on **FlashAttention**, [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)'s **FusedAttention**, and Context Parallelism-aware custom GEMM implementations — most of which are written in C++ and CUDA C++.
>
> A change that takes a few lines in PyTorch for small-scale experiments can trigger a chain of modifications across multiple dependency libraries when you need to maintain training speed at scale. The cost is fundamentally different from modifying code at the PyTorch level.

> **Q: How can I quickly verify the bias terms?**
>
> Compare the `model.safetensors.index.json` files of Llama-3.1-8B and gpt-oss-20b. You'll find that gpt-oss has additional `down_proj_bias` and `gate_up_proj_bias` entries in the MLP layers.
>
> ![Llama-3.1-8B model.safetensors.index.json](https://storage.googleapis.com/zenn-user-upload/fb7c6d4354cf-20251119.png)
> *[Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main?show_file_info=model.safetensors.index.json)*
>
> ![gpt-oss-20b model.safetensors.index.json](https://storage.googleapis.com/zenn-user-upload/04938c495895-20251119.png)
> *[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/tree/main?show_file_info=model.safetensors.index.json)*

## NGC

When looking into how to train gpt-oss, you'll find the [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/gpt_oss.html), which introduces the [25.07.gpt_oss](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.07.gpt_oss) container. It makes training seem straightforward — and it is, for small-scale finetuning.

However, for long-context training or continual pre-training, things are not so simple. This section describes how to set up the training environment using NGC.

### Extracting the Implementation

The following assumes work on a supercomputer using Singularity. Adapt the commands to your environment as needed.

First, create the `25.07.gpt_oss.def` file and build with Singularity:

```bash
# 25.07.gpt_oss.def
Bootstrap: docker
From: nvcr.io/nvidia/nemo:25.07.gpt_oss

%post
  pip install --no-cache-dir wandb transformers datasets jsonlines tqdm
```

I recommend building on local storage (e.g., `/scratch`) rather than Lustre or NFS to speed up the process.

```bash
cd /scratch
export SINGULARITY_TMPDIR=/scratch/tmp

singularity build --sandbox 25.07.gpt_oss 25.07.gpt_oss.def
```

We use a sandbox (not `.sif`) since we'll need to modify files inside the container.

The NeMo and Megatron-LM implementations inside this container differ from the tagged versions on GitHub, so we extract them for version control:

```bash
singularity shell --bind /path/to/your:/path/to/your 25.07.gpt_oss
Singularity>
```

Copy the implementations to an external path:

```bash
cp -R /opt/NeMo /path/to/your
cp -R /opt/megatron-lm /path/to/your
```

I've published the modified Megatron-LM on GitHub — feel free to use it:
[okoge-kaz/gpt-oss-megatron-lm](https://github.com/okoge-kaz/gpt-oss-megatron-lm)

> **Q: How did you locate the implementation inside the NeMo container?**
>
> The `dist-packages` directory didn't provide useful clues. I launched `ipython` and inspected the module paths:
> ```python
> In [1]: import importlib.metadata as md, megatron.core, pathlib
>    ...: print("version:", md.version("megatron-core"))
> version: 0.15.0rc0
>
> In [2]: print("module file :", megatron.core.__file__)
>    ...: print("module dir  :", pathlib.Path(megatron.core.__file__).parent)
> module file : /opt/megatron-lm/megatron/core/__init__.py
> module dir  : /opt/megatron-lm/megatron/core
> ```

### Applying Fixes

As noted in [this Pull Request](https://github.com/NVIDIA/Megatron-LM/pull/2038), the container's Megatron-LM implementation has a YaRN implementation that diverges from the HuggingFace version. The following files need to be patched:

```bash
Singularity> vim /opt/megatron-lm/megatron/core/models/common/embeddings/rope_utils.py
Singularity> vim /opt/megatron-lm/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
Singularity> vim /opt/megatron-lm/megatron/core/transformer/dot_product_attention.py
Singularity> vim /opt/megatron-lm/megatron/core/transformer/utils.py
```

The required diff is available here:
[Fix commit (GitHub)](https://github.com/okoge-kaz/gpt-oss-megatron-lm/commit/01b3824fe9d81b211b8aee6bfb35bd92169f8eb9)

> Note: There is divergence between the original PR's implementation and the container's code, so you also need to account for API changes unrelated to the PR itself. Arriving at the correct diff took considerable effort.

## NeMo

With the extracted NeMo under version control, we can start implementing.

### Current State

Training gpt-oss with NeMo requires converting HuggingFace-format checkpoints to NeMo format.
Moreover, the included tutorial (`tutorials/llm/gpt-oss/ticket-routing-lora/gpt-oss-lora.ipynb`) only covers LoRA SFT, and `nemo/collections/llm/recipes/gpt_oss_20b.py` doesn't support pretraining.

There's still a long way to go — let's tackle these one by one.

> **Q: Why is checkpoint conversion necessary?**
>
> Most models are distributed in HuggingFace format, which is easy to use but incompatible with high-performance training libraries like NVIDIA NeMo. Checkpoint conversion bridges this gap.
>
> The diagram below illustrates the flow: convert gpt-oss-20b from HF format to NeMo format, train with Japanese/domain data, and convert back to HF format for distribution.
>
> ![Checkpoint convert flow](https://storage.googleapis.com/zenn-user-upload/9d0ee43007a8-20251112.png)

### HF -> NeMo

The official documentation for the conversion script is frankly hard to follow. Here's a straightforward script:

```python
# experiments/ckpt-convert/hf-to-nemo/gpt-oss.py
import argparse
from nemo.collections import llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF GPT-OSS checkpoints to NeMo format.")
    parser.add_argument("--model-size", type=str, choices=["20B", "120B"], required=True)
    parser.add_argument("--hf-checkpoint-path", type=str, required=True)
    parser.add_argument("--nemo-output-path", type=str, required=True)
    args = parser.parse_args()

    if args.model_size == "20B":
        llm.import_ckpt(
            model=llm.GPTOSSModel(llm.GPTOSSConfig20B()),
            source="hf://" + args.hf_checkpoint_path,
            output_path=args.nemo_output_path,
        )
    elif args.model_size == "120B":
        llm.import_ckpt(
            model=llm.GPTOSSModel(llm.GPTOSSConfig120B()),
            source="hf://" + args.hf_checkpoint_path,
            output_path=args.nemo_output_path,
        )

    print(f"Conversion complete! NeMo checkpoint saved at {args.nemo_output_path}")
```

Usage:

```bash
HF_CHECKPOINT_PATH="/path/to/gpt-oss-20b"
NEMO_OUTPUT_PATH="/path/to/checkpoints/hf-to-nemo/gpt-oss-20B.nemo"
mkdir -p $(dirname ${NEMO_OUTPUT_PATH})

export NUMEXPR_MAX_THREADS=192

singularity exec \
  --nv \
  --bind /path/to:/path/to \
  --bind /tmp:/tmp \
  /path/to/25.07.gpt_oss.sif \
  python experiments/ckpt-convert/hf-to-nemo/gpt-oss.py \
    --model-size 20B \
    --hf-checkpoint-path ${HF_CHECKPOINT_PATH} \
    --nemo-output-path ${NEMO_OUTPUT_PATH}
```

### pretrain_recipe

The recipe files `gpt_oss_20b.py` and `gpt_oss_120b.py` only include a finetune recipe:

```python
@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "openai/gpt-oss-20b",
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = "lora",
    packed_sequence: bool = False,
) -> run.Partial:
```

We need to implement a pretrain recipe. Here's an example (can be simplified depending on your needs):

```python
@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    sequence_parallelism: bool = False,
    seq_length: int = 32768,
    global_batch_size: int = 256,
    micro_batch_size: int = 1,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    train_steps: int = 25000,
    warmup_steps: int = 1000,
    fp8: str = "",
    fn: Callable = pretrain,
) -> run.Partial:
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            tensor_parallelism=tensor_parallel_size,
            context_parallelism=context_parallel_size,
            pipeline_parallelism=pipeline_parallel_size,
            sequence_parallelism=sequence_parallelism,
            expert_parallel_size=expert_parallel_size,
            fp8=fp8,
            callbacks=[run.Config(TimingCallback, log_tokens_per_sec=True)],
        ),
        data=run.Config(MockDataModule, seq_length=seq_length,
                        global_batch_size=global_batch_size, micro_batch_size=micro_batch_size),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            train_steps=train_steps, warmup_steps=warmup_steps,
            max_lr=lr, min_lr=min_lr,
        ),
        resume=default_resume(),
    )
```

And the `trainer()` function:

```python
def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    context_parallelism: int = 2,
    expert_parallel_size: int = 4,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    fp8: str = "",
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        context_parallel_size=context_parallelism,
        expert_model_parallel_size=expert_parallel_size,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
        ),
    )

    precision = bf16_mixed()
    if fp8 == "current":
        precision = nemotron_h_bf16_with_fp8_current_scaling_mixed()
    elif fp8 == "blockwise":
        precision = bf16_with_fp8_subchannel_scaling_mixed()

    return run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=num_gpus_per_node,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=precision,
        strategy=strategy,
        callbacks=callbacks,
        use_distributed_sampler=False,
        val_check_interval=2000,
        log_every_n_steps=1,
    )
```

Details such as WandB logger callbacks, Megatron-LM-compatible checkpoint paths, and dataset configuration are omitted here for brevity.

### The Remaining Blocker

At this point, you'd hope training would work — but it doesn't.
The installed **FlashAttention** and **TransformerEngine** don't support **DotProductAttention** with gpt-oss's **learnable softmax**, so Context Parallelism is unavailable and the trainable context size is limited to around 8,192 tokens.

Setting `context_parallel_size > 1` results in:

```
ValueError: No dot product attention backend is available for the provided inputs.
            Please run with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to find out the reasons.
```

With debug flags enabled:

```
[DEBUG | DotProductAttention]: Disabling FlashAttention for softmax_type = learnable
[DEBUG | DotProductAttention]: Available backends = {
  FlashAttention=False, FusedAttention=False, UnfusedDotProductAttention=False
}
[DEBUG | DotProductAttention]: Selected backend = NoBackend
```

All three backends are disabled — hence the error.

### Updating TransformerEngine

Rather than implementing learnable softmax support from scratch, I investigated whether NVIDIA's TransformerEngine team had already done it. I found [TransformerEngine PR #2148](https://github.com/NVIDIA/TransformerEngine/pull/2148), which adds FusedAttention support for learnable softmax with `a2a` (all-to-all) Context Parallelism.

The fix: update TransformerEngine inside the sandbox to this version.

> **Q: Is updating TransformerEngine sufficient?**
>
> No. You also need to add the following to `GPTOSSConfig` in `nemo/collections/llm/gpt/model/gpt_oss.py`:
>
> ```python
>     attention_backend: AttnBackend = AttnBackend.fused
>     cp_comm_type: str = "a2a"
> ```
>
> This forces the use of FusedAttention (which supports learnable softmax) and sets the Context Parallel communication type to `a2a` instead of the default `p2p`.
>
> Without these settings:
> ```
> [DEBUG | DotProductAttention]: Disabling FusedAttention for context parallelism
>                                with softmax_type = learnable and cp_comm_type = p2p
> ```
>
> Source: [TransformerEngine utils.py#L721-L729](https://github.com/NVIDIA/TransformerEngine/blob/e7227af98070ebfcdb08b7f0a99bb87abe7b8532/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L721-L729)

### Updating cuDNN

Still not working. The debug output shows:

```
[DEBUG | DotProductAttention]: Disabling FusedAttention as no backend supports the provided input
```

Tracing this message to [fused_attn.cpp#L373-L376](https://github.com/NVIDIA/TransformerEngine/blob/e7227af98070ebfcdb08b7f0a99bb87abe7b8532/transformer_engine/common/fused_attn/fused_attn.cpp#L373-L376) reveals that cuDNN version 9.13.1+ is required, but the container ships with **9.13.0**.

The [PR description](https://github.com/NVIDIA/TransformerEngine/pull/2148) confirms:

> FusedAttention backend for FP16/BF16 and BSHD/SBHD: cuDNN 9.13.1+ and cudnn-frontend 1.14.1

![cuDNN Release](https://storage.googleapis.com/zenn-user-upload/a8e9a9f9f16b-20251104.png)

#### Solution

Since PyTorch and TransformerEngine load cuDNN as a shared library, we can simply bind a newer cuDNN into the container at runtime — no rebuild needed:

```bash
CUDNN_ROOT="/path/to/cudnn/cudnn-linux-x86_64-9.14.0.64_cuda12-archive"

singularity exec --nv .... \
  --bind ${CUDNN_ROOT}/lib:/usr/local/cudnn/lib64:ro \
  --bind ${CUDNN_ROOT}/include:/usr/local/cudnn/include:ro \
  /path/to/container/25.07.gpt_oss.fix.sif \
```

With all these changes, continual pre-training of GPT-OSS with **context length 32k** and **context parallel size = 4** is finally possible.

![Training Loss](https://storage.googleapis.com/zenn-user-upload/58f53e10c571-20251104.png)
*Training loss during continual pre-training*

> Don't forget to convert the modified sandbox to `.sif`:
> ```bash
> singularity build 25.07.gpt_oss.sif 25.07.gpt_oss
> ```

## Conclusion

This article covered the end-to-end process of training gpt-oss with NeMo. While LLM training may seem glamorous from the outside, the reality is that it's fundamentally a **software engineering challenge** — the implementation hurdles far outweigh what's typically described in papers.

As the [HuggingFace smol-training-playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction) puts it:

> The reality is messier, more iterative, and full of decisions that don't make it into the final paper.
