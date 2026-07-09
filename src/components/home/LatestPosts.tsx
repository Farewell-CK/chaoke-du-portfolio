"use client";

import { useTranslations, useLocale } from "next-intl";
import { Link } from "@/i18n/navigation";
import { ArrowUpRight, Calendar, Clock } from "lucide-react";

interface PostMeta {
  slug: string;
  title: { en: string; zh: string };
  date: string;
  excerpt: { en: string; zh: string };
  tags: string[];
  readingTime: number;
}

export default function LatestPosts({ posts }: { posts: PostMeta[] }) {
  const t = useTranslations("Home");
  const tBlog = useTranslations("Blog");
  const locale = useLocale() as "en" | "zh";

  if (posts.length === 0) {
    return null;
  }

  return (
    <section className="py-20 px-6 relative-z">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-end justify-between mb-12">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">
              {t("latestPosts")}
            </h2>
            <p className="text-muted">{t("latestPostsDesc")}</p>
          </div>
          <Link
            href="/blog"
            className="hidden md:flex items-center gap-1 text-sm text-accent hover:underline"
          >
            {t("viewAll")}
            <ArrowUpRight size={16} />
          </Link>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {posts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="group gradient-border rounded-2xl p-6 card-hover"
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
                  {post.readingTime} {tBlog("minRead")}
                </span>
              </div>

              <h3 className="text-lg font-bold mb-2 group-hover:text-accent transition-colors">
                {post.title[locale] || post.title.en}
              </h3>
              <p className="text-sm text-secondary line-clamp-3">
                {post.excerpt[locale] || post.excerpt.en}
              </p>

              <div className="flex flex-wrap gap-2 mt-4">
                {post.tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="text-xs px-2 py-0.5 rounded-md glass text-muted"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
