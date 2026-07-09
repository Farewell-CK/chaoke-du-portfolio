import { setRequestLocale } from "next-intl/server";
import { getTranslations } from "next-intl/server";
import { getAllProjects, type Project } from "@/lib/projects";
import ProjectCard from "@/components/projects/ProjectCard";

export default async function ProjectsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Projects");
  const projects = getAllProjects();

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-7xl mx-auto">
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-2 gradient-text">
            {t("title")}
          </h1>
          <p className="text-muted text-lg">{t("subtitle")}</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {projects.map((project) => (
            <ProjectCard
              key={project.slug}
              project={project}
              locale={locale as "en" | "zh"}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
