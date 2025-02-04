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
      "A framework for training LLM with PyTorch FSDP for every huggingface dense LLMs.",
    technologies: ["Python", "PyTorch"],
    imageUrl:
      "",
    codeUrl: "https://github.com/okoge-kaz/llm-recipes",
  },
];
