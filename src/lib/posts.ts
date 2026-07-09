import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import html from "remark-html";

const blogDirectory = path.join(process.cwd(), "content", "blog");

export interface BlogPost {
  slug: string;
  title: { en: string; zh: string };
  date: string;
  excerpt: { en: string; zh: string };
  tags: string[];
  contentHtml: string;
  readingTime: number;
}

export interface BlogPostMeta {
  slug: string;
  title: { en: string; zh: string };
  date: string;
  excerpt: { en: string; zh: string };
  tags: string[];
  readingTime: number;
}

function calculateReadingTime(content: string): number {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  return Math.ceil(words / wordsPerMinute);
}

export function getAllPostSlugs(): string[] {
  if (!fs.existsSync(blogDirectory)) {
    return [];
  }
  return fs
    .readdirSync(blogDirectory)
    .filter((file) => file.endsWith(".md"))
    .map((file) => file.replace(/\.md$/, ""));
}

export function getPostBySlug(slug: string): BlogPost | null {
  const fullPath = path.join(blogDirectory, `${slug}.md`);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, "utf8");
  const { data, content } = matter(fileContents);

  const processedContent = remark().use(html).processSync(content);
  const contentHtml = processedContent.toString();
  const readingTime = calculateReadingTime(content);

  return {
    slug,
    title: data.title || { en: slug, zh: slug },
    date: data.date || new Date().toISOString(),
    excerpt: data.excerpt || { en: "", zh: "" },
    tags: data.tags || [],
    contentHtml,
    readingTime,
  };
}

export function getAllPosts(): BlogPostMeta[] {
  const slugs = getAllPostSlugs();

  return slugs
    .map((slug) => {
      const post = getPostBySlug(slug);
      if (!post) return null;
      return {
        slug: post.slug,
        title: post.title,
        date: post.date,
        excerpt: post.excerpt,
        tags: post.tags,
        readingTime: post.readingTime,
      };
    })
    .filter((post): post is BlogPostMeta => post !== null)
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getLatestPosts(limit: number = 3): BlogPostMeta[] {
  return getAllPosts().slice(0, limit);
}
