export interface News {
  date: string;
  title: string;
  description: string;
  link?: string;
}

export const newsData: News[] = [
  // If you don't want to show news, just make the array empty.
  /*{
    date: "March 2024",
    title: "Paper accepted at ICML 2024",
    description: "Our work on causal discovery in time series data has been accepted at ICML 2024.",
    link: "https://icml.cc/",
  }*/
  {
    date: "August 2024",
    title: "Google Cloud Next '24 Tokyo Talk",
    description:
      "I gave a talk at Google Cloud Next '24 Tokyo on the topic of 'How to use Google Cluster Toolkit and real use-case'.",
    link: "https://cloudonair.withgoogle.com/events/next-tokyo-24?talk=d2-inf-03"
  },
  {
    date: "March 2024",
    title: "NLP 2024 workshop talk",
    description:
      "I gave a talk at the NLP 2024 workshop on the topic of 'Distributed Training Technologies for Natural Language Processing'.",
    link: "https://sites.google.com/view/llm-discussion-nlp2024-ws"
  },
];
