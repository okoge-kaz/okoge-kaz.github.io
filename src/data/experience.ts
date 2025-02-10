export interface Experience {
  date: string;
  title: string;
  company: string;
  description?: string;
  advisor?: string;
  manager?: string;
  companyUrl?: string;
}

export const experienceData: Experience[] = [
  {
    date: "Apr 2024 - Present",
    title: "Research Intern",
    company: "SB Intuitions",
    description:
      "Worked on developing frameworks on training large language models.",
    manager: "Sho Takase",
    companyUrl: "https://www.sbintuitions.co.jp/",
  },
  {
    date: "Oct 2023 - Present",
    title: "Research Intern",
    company: "AIST (National Institute of Advanced Industrial Science and Technology)",
    description:
      "I am involved in selecting and maintaining pre-training and post-training libraries, managing experiments, and setting up experimental environments to develop a Japanese LLM with competitive performance. This initiative, known as the Swallow Project (https://swallow-llm.github.io/index.en.html), has contributed to the development of non-English LLMs by achieving top performance among open Japanese models as of December 2023. As a core contributor to the project, I have been broadly involved in all aspects of the training process—from procuring computational resources and maintaining the Environment Module to creating synthetic data.",
    manager: "Hiroya Takamura",
    companyUrl: "https://www.airc.aist.go.jp/en/",
  },
  {
    date: "Feb 2023 - Present",
    title: "Research Intern",
    company: "Turing",
    description:
      "Worked on developing frameworks on training vision-language models and large language models.",
    manager: "Yu Yamaguchi",
    companyUrl: "https://tur.ing/en",
  },
  {
    date: "Apr 2024 - Dec 2024",
    title: "Intern",
    company: "Sakana AI",
    description:
      "Worked on deploying and maintaining H100 cluster for research and development of large language models.",
    manager: "Takuya Akiba",
    companyUrl: "https://sakana.ai/",
  },
  {
    date: "Oct 2023 - Feb 2024",
    title: "Research Intern",
    company: "Kotoba Technologies",
    description:
      "Worked on developing LLM training library and working on training large language models. I developed Mamba training library on Dec 2023 when huggingface didn't support mamba training at that time.",
    manager: "Noriyuki Kojima",
    companyUrl: "https://www.kotoba.tech/en/home",
  },
  {
    date: "Summer 2023",
    title: "Intern",
    company: "Preferred Networks, Inc.",
    description:
      "Developed ImageRecognition System for Real-world Applications",
    companyUrl: "https://www.preferred.jp/en/",
  },
];
