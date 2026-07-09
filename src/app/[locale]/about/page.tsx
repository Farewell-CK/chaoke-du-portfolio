import { setRequestLocale, getTranslations } from "next-intl/server";
import { Code2, BookOpen, Brain, Database, Server, Wrench } from "lucide-react";

export default async function AboutPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("About");
  const loc = locale as "en" | "zh";

  const skills = [
    { name: "Python", icon: Code2, category: "Language" },
    { name: "TypeScript", icon: Code2, category: "Language" },
    { name: "React / Next.js", icon: Code2, category: "Frontend" },
    { name: "FastAPI", icon: Server, category: "Backend" },
    { name: "PostgreSQL", icon: Database, category: "Database" },
    { name: "ChromaDB", icon: Database, category: "Database" },
    { name: "Docker", icon: Wrench, category: "DevOps" },
    { name: "AI / LLMs", icon: Brain, category: "AI" },
  ];

  const timeline = loc === "zh"
    ? [
        { period: "2024 - Present", title: "研究生在读", desc: "研究 AI 辅助科研方向，构建论文阅读工具" },
        { period: "2020 - 2024", title: "本科", desc: "计算机科学专业" },
      ]
    : [
        { period: "2024 - Present", title: "Graduate Student", desc: "Researching AI-assisted science, building paper reading tools" },
        { period: "2020 - 2024", title: "Undergraduate", desc: "Computer Science" },
      ];

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl md:text-5xl font-bold mb-8 gradient-text">
          {t("title")}
        </h1>

        <div className="glass rounded-2xl p-8 mb-12 gradient-border">
          <p className="text-lg text-secondary leading-relaxed">{t("bio")}</p>
        </div>

        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Code2 className="text-accent" size={24} />
          {t("skills")}
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          {skills.map((skill) => (
            <div
              key={skill.name}
              className="glass rounded-xl p-4 text-center card-hover"
            >
              <skill.icon className="text-accent mx-auto mb-2" size={24} />
              <p className="text-sm font-medium">{skill.name}</p>
              <p className="text-xs text-muted mt-1">{skill.category}</p>
            </div>
          ))}
        </div>

        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <BookOpen className="text-accent" size={24} />
          {t("education")}
        </h2>
        <div className="space-y-4">
          {timeline.map((item, i) => (
            <div
              key={i}
              className="glass rounded-xl p-6 gradient-border flex flex-col md:flex-row md:items-center gap-4"
            >
              <span className="text-sm text-accent font-mono shrink-0">
                {item.period}
              </span>
              <div>
                <h3 className="font-bold">{item.title}</h3>
                <p className="text-sm text-muted">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
