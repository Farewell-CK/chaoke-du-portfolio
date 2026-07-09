import { NextResponse } from "next/server";
import { getAllProjects } from "@/lib/projects";
import { getAllPosts } from "@/lib/posts";
import { resources } from "@/lib/resources";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q")?.toLowerCase() || "";

  if (!query || query.length < 2) {
    return NextResponse.json({ results: [] });
  }

  const results: Array<{
    type: string;
    title: string;
    description: string;
    url: string;
    tags: string[];
  }> = [];

  const projects = getAllProjects();
  for (const p of projects) {
    const title = p.title.en + " " + p.title.zh;
    const desc = p.description.en + " " + p.description.zh;
    const tags = p.tags.join(" ");
    if (
      title.toLowerCase().includes(query) ||
      desc.toLowerCase().includes(query) ||
      tags.toLowerCase().includes(query)
    ) {
      results.push({
        type: "project",
        title: p.title.en,
        description: p.description.en,
        url: `/projects/${p.slug}`,
        tags: p.tags,
      });
    }
  }

  const posts = getAllPosts();
  for (const p of posts) {
    const title = p.title.en + " " + (p.title.zh || "");
    const excerpt = p.excerpt.en + " " + (p.excerpt.zh || "");
    const tags = p.tags.join(" ");
    if (
      title.toLowerCase().includes(query) ||
      excerpt.toLowerCase().includes(query) ||
      tags.toLowerCase().includes(query)
    ) {
      results.push({
        type: "blog",
        title: p.title.en,
        description: p.excerpt.en,
        url: `/blog/${p.slug}`,
        tags: p.tags,
      });
    }
  }

  for (const r of resources) {
    const title = r.title.en + " " + (r.title.zh || "");
    const desc = r.description.en + " " + (r.description.zh || "");
    const tags = r.tags.join(" ");
    if (
      title.toLowerCase().includes(query) ||
      desc.toLowerCase().includes(query) ||
      tags.toLowerCase().includes(query)
    ) {
      results.push({
        type: "resource",
        title: r.title.en,
        description: r.description.en,
        url: r.url,
        tags: r.tags,
      });
    }
  }

  return NextResponse.json({ results });
}
