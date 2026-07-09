"use client";

import { useTranslations, useLocale } from "next-intl";
import { Link } from "@/i18n/navigation";
import { Calendar, Clock, ArrowUpRight } from "lucide-react";

interface PostMeta {
  slug: string;
  title: { en: string; zh: string };
  date: string;
  excerpt: { en: string; zh: string };
  tags: string[];
  readingTime: number;
}

export default function BlogList({ posts }: { posts: PostMeta[] }) {
  const t = useTranslations("Blog");
  const locale = useLocale() as "en" | "zh";

  if (posts.length === 0) {
    return (
      <div className="text-center py-20 text-muted">
        <p className="text-lg">No posts yet. Check back soon!</p>
      </div>
    );
  }

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
      {posts.map((post) => (
        <Link
          key={post.slug}
          href={`/blog/${post.slug}`}
          className="group gradient-border rounded-2xl p-6 card-hover flex flex-col"
        >
          <div className="flex items-center gap-3 text-xs text-muted mb-3">
            <span className="flex items-center gap-1">
              <Calendar size={12} />
              {new Date(post.date).toLocaleDateString(
                locale === "zh" ? "zh-CN" : "en-US",
                { year: "numeric", month: "short", day: "numeric" }
              )}
            </span>
            <span className="flex items-center gap-1">
              <Clock size={12} />
              {post.readingTime} {t("minRead")}
            </span>
          </div>

          <h3 className="text-lg font-bold mb-2 group-hover:text-accent transition-colors">
            {post.title[locale] || post.title.en}
          </h3>
          <p className="text-sm text-secondary line-clamp-3 flex-1">
            {post.excerpt[locale] || post.excerpt.en}
          </p>

          <div className="flex items-center justify-between mt-4">
            <div className="flex flex-wrap gap-2">
              {post.tags.slice(0, 2).map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-2 py-0.5 rounded-md glass text-muted"
                >
                  {tag}
                </span>
              ))}
            </div>
            <ArrowUpRight
              size={16}
              className="text-accent transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
            />
          </div>
        </Link>
      ))}
    </div>
  );
}
