import { setRequestLocale, getTranslations } from "next-intl/server";
import { resources, type Resource } from "@/lib/resources";
import { Book, FileText, Wrench, GraduationCap, ExternalLink } from "lucide-react";

const categoryIcons: Record<string, typeof Book> = {
  book: Book,
  paper: FileText,
  tool: Wrench,
  course: GraduationCap,
};

export default async function ResourcesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Resources");
  const loc = locale as "en" | "zh";

  const categories = [
    { key: "all", label: loc === "zh" ? "全部" : "All" },
    { key: "book", label: loc === "zh" ? "书籍" : "Books" },
    { key: "paper", label: loc === "zh" ? "论文" : "Papers" },
    { key: "tool", label: loc === "zh" ? "工具" : "Tools" },
    { key: "course", label: loc === "zh" ? "课程" : "Courses" },
  ];

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl md:text-5xl font-bold mb-2 gradient-text">
          {t("title")}
        </h1>
        <p className="text-muted text-lg mb-8">{t("subtitle")}</p>

        <div className="flex flex-wrap gap-2 mb-8">
          {categories.map((cat) => (
            <span
              key={cat.key}
              className="text-sm px-4 py-2 rounded-full glass text-secondary cursor-default"
            >
              {cat.label}
            </span>
          ))}
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {resources.map((resource) => {
            const Icon = categoryIcons[resource.category] || Book;
            return (
              <a
                key={resource.id}
                href={resource.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group glass rounded-2xl p-6 card-hover gradient-border"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="w-12 h-12 rounded-xl glass flex items-center justify-center">
                    <Icon className="text-accent" size={22} />
                  </div>
                  <ExternalLink
                    size={16}
                    className="text-muted group-hover:text-accent transition-colors"
                  />
                </div>

                <h3 className="text-lg font-bold mb-2 group-hover:text-accent transition-colors">
                  {resource.title[loc] || resource.title.en}
                </h3>
                <p className="text-sm text-secondary line-clamp-2 mb-4">
                  {resource.description[loc] || resource.description.en}
                </p>

                <div className="flex flex-wrap gap-2">
                  {resource.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-0.5 rounded-md glass text-muted"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </a>
            );
          })}
        </div>
      </div>
    </div>
  );
}
