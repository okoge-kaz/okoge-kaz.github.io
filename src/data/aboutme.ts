export interface AboutMe {
  name: string;
  title: string;
  institution: string;
  description: string;
  email: string;
  imageUrl?: string;
  blogUrl?: string;
  cvUrl?: string;
  googleScholarUrl?: string;
  twitterUsername?: string;
  githubUsername?: string;
  linkedinUsername?: string;
  funDescription?: string; // Gets placed in the left sidebar
  secretDescription?: string; // Gets placed in the bottom
  altName?: string;
  institutionUrl?: string;
}

export const aboutMe: AboutMe = {
  name: "Kazuki Fujii",
  title: "Master's Student",
  institution: "Institute of Science Tokyo",
  // Note that links work in the description
  description:
    "I'm a first-year <a href='https://www.isct.ac.jp/en'>Master's student</a> working at the intersection of HPC and Machine Learning. My research focuses on distributed training of large models and low-precision training. Also, I am in charge of the maintenance of Pre-training Library for LLM and conducting experiments on training LLMs.\n\nI am a core contributor of <a href='https://swallow-llm.github.io/index.en.html'>Swallow Project</a> which is a Japanese LLM development initiative.\n\nI am currently looking for research internships in the field of HPC and Machine Learning.",
  email: "kazuki.fujii@rio.scrc.iir.isct.ac.jp",
  imageUrl:
    "",
  googleScholarUrl: "https://scholar.google.co.jp/citations?user=jHXLs2wAAAAJ",
  githubUsername: "okoge-kaz",
  linkedinUsername: "kazuki-fujii",
  twitterUsername: "okoge_kaz",
  blogUrl: "https://zenn.dev/kaz20",
  cvUrl: "",
  institutionUrl: "https://www.isct.ac.jp/en",
  // altName: "",
  // secretDescription: "I like dogs.",
};
