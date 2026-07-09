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
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 gradient-text">
            {t("title")}
          </h1>
          <p className="text-muted text-lg">{t("subtitle")}</p>
        </div>
        <BlogList posts={posts} />
      </div>
    </div>
  );
}
