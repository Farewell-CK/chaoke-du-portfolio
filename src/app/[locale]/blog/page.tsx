import { setRequestLocale, getTranslations } from "next-intl/server";
import { getAllPosts } from "@/lib/posts";
import BlogList from "@/components/blog/BlogList";

export default async function BlogPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Blog");
  const posts = getAllPosts();

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-7xl mx-auto">
        <div className="mb-16">
          <div className="flex items-center gap-3 mb-4">
            <span className="h-px w-12 bg-gradient-to-r from-transparent to-accent/50" />
            <span className="text-xs uppercase tracking-[0.3em] text-accent font-mono">Blog</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold mb-4 gradient-text">
            {t("title")}
          </h1>
          <p className="text-muted text-lg max-w-2xl">{t("subtitle")}</p>
        </div>
        <BlogList posts={posts} />
      </div>
    </div>
  );
}
