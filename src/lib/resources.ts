export interface Resource {
  id: string;
  title: { en: string; zh: string };
  description: { en: string; zh: string };
  url: string;
  category: "book" | "paper" | "tool" | "course";
  tags: string[];
}

export const resources: Resource[] = [
  {
    id: "1",
    title: {
      en: "Attention Is All You Need",
      zh: "Attention Is All You Need",
    },
    description: {
      en: "The foundational paper introducing the Transformer architecture.",
      zh: "引入 Transformer 架构的基础论文。",
    },
    url: "https://arxiv.org/abs/1706.03762",
    category: "paper",
    tags: ["AI", "NLP", "Transformer"],
  },
  {
    id: "2",
    title: {
      en: "Deep Learning",
      zh: "深度学习",
    },
    description: {
      en: "Ian Goodfellow's comprehensive textbook on deep learning.",
      zh: "Ian Goodfellow 的深度学习综合教材。",
    },
    url: "https://www.deeplearningbook.org/",
    category: "book",
    tags: ["AI", "Deep Learning"],
  },
  {
    id: "3",
    title: {
      en: "FastAPI",
      zh: "FastAPI",
    },
    description: {
      en: "Modern, fast web framework for building APIs with Python.",
      zh: "现代化、高性能的 Python API 框架。",
    },
    url: "https://fastapi.tiangolo.com/",
    category: "tool",
    tags: ["Python", "Backend", "API"],
  },
  {
    id: "4",
    title: {
      en: "CS224N: Natural Language Processing with Deep Learning",
      zh: "CS224N: 深度学习自然语言处理",
    },
    description: {
      en: "Stanford's course on NLP with deep learning.",
      zh: "斯坦福大学深度学习自然语言处理课程。",
    },
    url: "https://web.stanford.edu/class/cs224n/",
    category: "course",
    tags: ["NLP", "Stanford", "Course"],
  },
];

export function getResourcesByCategory(category?: string): Resource[] {
  if (!category || category === "all") return resources;
  return resources.filter((r) => r.category === category);
}

export function getAllCategories(): string[] {
  return ["all", "book", "paper", "tool", "course"];
}
