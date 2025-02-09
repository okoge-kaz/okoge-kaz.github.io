export interface Portfolio {
  title: string;
  description: string;
  technologies?: string[];
  imageUrl?: string;
  projectUrl?: string;
  codeUrl?: string;
}

export const portfolioData: Portfolio[] = [
  // Example entry
  {
    title: "vlm-recipes: VLM training Framework",
    description:
      "A framework for training vision-language models with PyTorch FSDP.",
    technologies: ["Python", "PyTorch"],
    imageUrl:
      "",
    codeUrl: "https://github.com/turingmotors/vlm-recipes",
  },
  {
    title: "moe-recipes: Mixture of Experts LLM training Framework",
    description:
      "A framework for training MoE with DeepSpeed.",
    technologies: ["Python", "PyTorch"],
    imageUrl:
      "",
    codeUrl: "https://github.com/okoge-kaz/moe-recipes",
  },
  {
    title: "llm-recipes: LLM continual pre-training & post-training Framework",
    description:
      "As of January 2024, since [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) did not support training [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) models, I built upon Meta’s llama-recipes (now known as [llama-cookbook](https://github.com/meta-llama/llama-cookbook)) to develop a library that enables the training of non-Llama models. I modified the DataLoader to handle training at a 100B-token scale, integrated [wandb](https://wandb.ai/site/) logging, and implemented additional essential training features such as learning rate scheduling. The resulting library, llm-recipes, supports continual pre-training, supervised fine-tuning (SFT), and DPO. This work was submitted to the [SC24 TPC workshop](https://tpc.dev/tpc-workshop-at-sc24/) and accepted.",
    technologies: ["Python", "PyTorch"],
    imageUrl:
      "",
    codeUrl: "https://github.com/okoge-kaz/llm-recipes",
  },
  {
    title: "kotomamba: State Space Model training Framework",
    description:
      "As of December 2023, even popular libraries like Hugging Face Transformers did not support Mamba. To enable both from-scratch training and continual pre-training for Mamba models, I independently developed kotomamba—a distributed training library built on PyTorch FSDP.",
    technologies: ["Python", "PyTorch"],
    imageUrl:
      "",
    codeUrl: "https://github.com/kotoba-tech/kotomamba",
  }
];
