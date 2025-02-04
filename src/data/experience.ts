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
