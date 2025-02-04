export interface Education {
  year: string;
  institution: string;
  degree: string;
  advisor?: string;
  thesis?: string;
  thesisUrl?: string;
}

export const educationData: Education[] = [
  // If you don't want to show education, just make the array empty.
  {
    year: "2024—Present",
    institution: "Institute of Science Tokyo",
    degree: "Master in Computer Science",
    advisor: "Prof. Jun Sakuma and Prof. Rio Yokota",
  },
  {
    year: "2020—2024",
    institution: "Tokyo Institute of Technology",
    degree: "B.S. in Computer Science",
    thesis: "",
    // Optional links to thesis
    // thesisUrl: "https://dspace.mit.edu/handle/1721.1/149111"
  },
];
