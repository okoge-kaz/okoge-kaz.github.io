export interface Publication {
  year: string;
  conference: string;
  title: string;
  authors: string;
  paperUrl?: string;
  codeUrl?: string;
  bibtex?: string;
  tldr?: string;
  imageUrl?: string;
  award?: string;
}

export const publicationData: Publication[] = [
  // If you don't want to show publications, just make the array empty.
  {
    year: "2024",
    conference: "CVPR (workshop)",
    title: "Heron-Bench: A Benchmark for Evaluating Vision Language Models in Japanese",
    authors: "Yuichi Inoue, Kento Sasaki, Yuma Ochi, Kazuki Fujii, Kotaro Tanahashi, Yu Yamaguchi",
    paperUrl: "https://arxiv.org/abs/2404.07824",
    // if you have an image in public/images, you can use it like this:
    // imageUrl: "/images/publication-image.jpg"
  },
  {
    year: "2024",
    conference: "COLM",
    title: "Building a Large Japanese Web Corpus for Large Language Models",
    authors: "Naoaki Okazaki, Kakeru Hattori, Hirai Shota, Hiroki Iida, Masanari Ohi, Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Rio Yokota, Sakae Mizuki",
    paperUrl: "https://arxiv.org/abs/2404.17733",
  },
  {
    year: "2024",
    conference: "COLM",
    title: "Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities",
    authors: "Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, Naoaki Okazaki",
    paperUrl: "https://arxiv.org/abs/2404.17790",
    codeUrl: "https://github.com/rioyokotalab/Megatron-Llama2",
  },
];
