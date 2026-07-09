import { setRequestLocale, getTranslations } from "next-intl/server";
import { notFound } from "next/navigation";
import { getPostBySlug, getAllPostSlugs } from "@/lib/posts";
import { Link } from "@/i18n/navigation";
import { ArrowLeft, Calendar, Clock } from "lucide-react";

export function generateStaticParams() {
  return getAllPostSlugs().map((slug) => ({ slug }));
}

export default async function BlogDetailPage({
  params,
}: {
  params: Promise<{ locale: string; slug: string }>;
}) {
  const { locale, slug } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Blog");
  const tCommon = await getTranslations("Common");

  const post = getPostBySlug(slug);
  if (!post) {
    notFound();
  }

  const loc = locale as "en" | "zh";

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-3xl mx-auto">
        <Link
          href="/blog"
          className="inline-flex items-center gap-1 text-sm text-muted hover:text-accent transition-colors mb-8"
        >
          <ArrowLeft size={16} />
          {tCommon("back")}
        </Link>

        <article>
          <h1 className="text-3xl md:text-5xl font-bold mb-4 leading-tight">
            {post.title[loc] || post.title.en}
          </h1>

          <div className="flex items-center gap-4 text-sm text-muted mb-8">
            <span className="flex items-center gap-1">
              <Calendar size={14} />
              {new Date(post.date).toLocaleDateString(
                loc === "zh" ? "zh-CN" : "en-US",
                { year: "numeric", month: "long", day: "numeric" }
              )}
            </span>
            <span className="flex items-center gap-1">
              <Clock size={14} />
              {post.readingTime} {t("minRead")}
            </span>
          </div>

          {post.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-8">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-sm px-3 py-1 rounded-lg glass text-secondary"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          <div
            className="prose prose-invert max-w-none
              [&_h2]:text-2xl [&_h2]:font-bold [&_h2]:mt-8 [&_h2]:mb-4 [&_h2]:text-foreground
              [&_h3]:text-xl [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-3 [&_h3]:text-foreground
              [&_p]:text-secondary [&_p]:leading-relaxed [&_p]:mb-4
              [&_ul]:list-disc [&_ul]:pl-6 [&_ul]:mb-4 [&_ul]:text-secondary [&_ul]:leading-relaxed
              [&_ol]:list-decimal [&_ol]:pl-6 [&_ol]:mb-4 [&_ol]:text-secondary [&_ol]:leading-relaxed
              [&_li]:mb-1
              [&_code]:bg-card [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-accent [&_code]:text-sm [&_code]:font-mono
              [&_pre]:glass [&_pre]:rounded-xl [&_pre]:p-4 [&_pre]:mb-4 [&_pre]:overflow-x-auto
              [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-foreground
              [&_blockquote]:border-l-2 [&_blockquote]:border-accent [&_blockquote]:pl-4 [&_blockquote]:text-muted [&_blockquote]:italic
              [&_a]:text-accent [&_a]:underline [&_a]:hover:text-accent-violet
              [&_hr]:border-border [&_hr]:my-8
              [&_strong]:text-foreground [&_strong]:font-semibold
              [&_table]:w-full [&_table]:border-collapse [&_table]:mb-4 [&_table]:text-sm
              [&_thead]:border-b [&_thead]:border-border
              [&_th]:text-left [&_th]:font-semibold [&_th]:text-foreground [&_th]:px-4 [&_th]:py-2 [&_th]:border [&_th]:border-border
              [&_td]:px-4 [&_td]:py-2 [&_td]:border [&_td]:border-border [&_td]:text-secondary
              [&_tr]:border-border [&_tr:hover]:bg-card/50"
            dangerouslySetInnerHTML={{ __html: post.contentHtml }}
          />
        </article>
      </div>
    </div>
  );
}
