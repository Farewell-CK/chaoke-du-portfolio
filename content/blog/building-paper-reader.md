---
title:
  en: "Building an AI-Powered Paper Reading Tool"
  zh: "构建 AI 驱动的论文带读工具"
date: "2026-06-26"
excerpt:
  en: "My journey building PaperReader, an AI assistant that reads academic papers with you."
  zh: "我构建 PaperReader 的旅程——一个陪你读论文的 AI 助手。"
tags: ["AI", "Research", "FastAPI", "React"]
---

## Why I Built This

As a graduate student, I read dozens of papers every week. The process is often tedious - dense text, unfamiliar jargon, and complex methodologies that take hours to unpack.

I wanted a tool that could **read with me**, not just summarize for me. Something that guides me through a paper paragraph by paragraph, explains the hard parts, and answers my questions in context.

## The Three Reading Modes

### 1. Paragraph-by-Paragraph Dialogue

The AI presents each section, pauses for your questions, and only continues when you're ready. This is perfect for deep, careful reading.

### 2. Auto-Scroll with Commentary

For faster skimming, the AI scrolls through the paper while generating real-time commentary. You can pause anytime to ask questions.

### 3. Chapter Summaries

After each chapter, the AI generates a summary and key questions for deeper exploration.

## Tech Stack

- **Frontend**: React + TypeScript + TailwindCSS
- **Backend**: Python FastAPI
- **Database**: PostgreSQL + ChromaDB (vector search)
- **AI**: OpenAI GPT-5.5
- **CLI**: Python Typer (for agent integration)

## What's Next

I'm currently integrating Feishu knowledge base support, so your paper notes can sync with your research wiki automatically. Stay tuned!

---

*This is a sample blog post. Replace it with your own content in `content/blog/`.*
