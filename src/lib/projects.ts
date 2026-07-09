export interface Project {
  slug: string;
  title: { en: string; zh: string };
  description: { en: string; zh: string };
  longDescription?: { en: string; zh: string };
  type: "builtin" | "standalone";
  tags: string[];
  url?: string;
  github?: string;
  icon: string;
  featured: boolean;
  status: "active" | "development" | "archived";
}

export const projects: Project[] = [
  {
    slug: "ai-image-studio",
    title: {
      en: "AI Image Studio",
      zh: "AI 图像工作室",
    },
    description: {
      en: "A browser-based image generation and editing workspace for prompt-driven visual creation.",
      zh: "面向提示词视觉创作的浏览器图像生成与编辑工作台。",
    },
    longDescription: {
      en: "AI Image Studio is a focused visual creation tool for experimenting with image prompts, layouts, and generation workflows directly in the browser.",
      zh: "AI 图像工作室是一个面向视觉创作的工具，用于在浏览器中实验图像提示词、布局和生成工作流。",
    },
    type: "standalone",
    tags: ["Next.js", "React", "AI", "Image"],
    url: "/tools/ai-image-studio/",
    icon: "Image",
    featured: true,
    status: "active",
  },
  {
    slug: "music-visualizer",
    title: {
      en: "Music Visualizer",
      zh: "音乐可视化器",
    },
    description: {
      en: "An interactive audio visualization experience for exploring sound through motion and color.",
      zh: "通过动态与色彩探索声音的交互式音乐可视化体验。",
    },
    longDescription: {
      en: "Music Visualizer turns audio input and playback into responsive visual scenes designed for experimentation, presentation, and creative exploration.",
      zh: "音乐可视化器将音频输入和播放转化为响应式视觉场景，适合实验、展示和创意探索。",
    },
    type: "standalone",
    tags: ["Next.js", "Audio", "Visualization", "Interactive"],
    url: "/tools/music-visualizer/",
    icon: "Music",
    featured: true,
    status: "active",
  },
  {
    slug: "schemaforge",
    title: {
      en: "SchemaForge",
      zh: "SchemaForge 数据库结构设计器",
    },
    description: {
      en: "An AI-assisted database schema design tool that turns product scenarios into structured database models.",
      zh: "AI 辅助数据库结构设计工具，把业务场景转成结构化数据库模型。",
    },
    longDescription: {
      en: "SchemaForge helps design relational data models, inspect entity relationships, and export architecture-ready schema drafts from natural product requirements.",
      zh: "SchemaForge 帮助从自然语言产品需求出发，设计关系型数据模型、检查实体关系，并导出可用于架构设计的 schema 草稿。",
    },
    type: "standalone",
    tags: ["Vite", "React", "Database", "Schema"],
    url: "/tools/schemaforge/",
    icon: "Database",
    featured: true,
    status: "active",
  },
  {
    slug: "solution-architect",
    title: {
      en: "Solution Architect",
      zh: "解决方案架构师",
    },
    description: {
      en: "A visual architecture design tool for drafting systems, flows, and implementation plans.",
      zh: "用于绘制系统、流程和实施方案的可视化架构设计工具。",
    },
    longDescription: {
      en: "Solution Architect provides a visual workspace for mapping product requirements into system components, data flows, and technical implementation plans.",
      zh: "解决方案架构师提供可视化工作台，用于把产品需求映射为系统组件、数据流和技术实施方案。",
    },
    type: "standalone",
    tags: ["Vite", "React", "Architecture", "Flow"],
    url: "/tools/solution-architect/",
    icon: "Network",
    featured: true,
    status: "active",
  },
  {
    slug: "competitor-ai",
    title: {
      en: "CompetitorAI",
      zh: "CompetitorAI 竞品分析平台",
    },
    description: {
      en: "An AI-powered competitive intelligence platform for discovery, analysis, comparison, and reporting.",
      zh: "用于竞品发现、分析、对比和报告生成的 AI 竞品情报平台。",
    },
    longDescription: {
      en: "CompetitorAI combines search, scraping, LLM analysis, SWOT generation, comparison charts, and report export. It requires backend API keys, so it is listed here as a project rather than directly embedded as a static tool.",
      zh: "CompetitorAI 结合搜索、网页抓取、LLM 分析、SWOT 生成、对比图表和报告导出。它需要后端 API key，因此这里先作为项目展示，而不是直接作为静态工具嵌入。",
    },
    type: "standalone",
    tags: ["React", "Express", "AI", "Analytics"],
    icon: "BarChart3",
    featured: false,
    status: "development",
  },
  {
    slug: "paper-reader",
    title: {
      en: "PaperReader - AI Paper Reading Assistant",
      zh: "PaperReader - AI 论文带读工具",
    },
    description: {
      en: "An AI-powered tool that reads academic papers with you, paragraph by paragraph.",
      zh: "AI 驱动的论文带读工具，逐段陪你读论文。",
    },
    longDescription: {
      en: "PaperReader is a comprehensive research tool that helps you read and understand academic papers. It features three reading modes (paragraph-by-paragraph dialogue, auto-scroll with commentary, and chapter summaries), AI-powered Q&A, annotation and note-taking, vector search, and Feishu knowledge base integration. Built with React, FastAPI, PostgreSQL, and GPT-5.5.",
      zh: "PaperReader 是一个综合性的科研工具，帮助你阅读和理解学术论文。它具有三种带读模式（逐段对话、自动滚动讲解、章节总结）、AI 问答、标注笔记、向量搜索和飞书知识库集成。基于 React、FastAPI、PostgreSQL 和 GPT-5.5 构建。",
    },
    type: "standalone",
    tags: ["AI", "FastAPI", "React", "PostgreSQL", "GPT-5.5"],
    url: "/app/paper-reader",
    github: "https://github.com",
    icon: "BookOpen",
    featured: true,
    status: "development",
  },
  {
    slug: "portfolio",
    title: {
      en: "This Website",
      zh: "本网站",
    },
    description: {
      en: "My personal portfolio built with Next.js, featuring i18n, Markdown blog, and more.",
      zh: "使用 Next.js 构建的个人作品集网站，支持国际化、Markdown 博客等。",
    },
    type: "builtin",
    tags: ["Next.js", "TypeScript", "TailwindCSS", "Docker"],
    url: "/",
    icon: "Code2",
    featured: true,
    status: "active",
  },
];

export function getFeaturedProjects(): Project[] {
  return projects.filter((p) => p.featured);
}

export function getAllProjects(): Project[] {
  return projects;
}

export function getProjectBySlug(slug: string): Project | undefined {
  return projects.find((p) => p.slug === slug);
}
