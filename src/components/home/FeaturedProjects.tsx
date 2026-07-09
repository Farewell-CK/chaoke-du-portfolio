"use client";

import { useTranslations, useLocale } from "next-intl";
import { Link } from "@/i18n/navigation";
import { getFeaturedProjects, type Project } from "@/lib/projects";
import { ArrowUpRight, Wrench, ExternalLink } from "lucide-react";

export default function FeaturedProjects() {
  const t = useTranslations("Home");
  const tProjects = useTranslations("Projects");
  const locale = useLocale() as "en" | "zh";
  const featuredProjects = getFeaturedProjects();

  return (
    <section className="py-20 px-6 relative-z">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-end justify-between mb-12">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-2">
              {t("featuredProjects")}
            </h2>
            <p className="text-muted">{t("featuredProjectsDesc")}</p>
          </div>
          <Link
            href="/projects"
            className="hidden md:flex items-center gap-1 text-sm text-accent hover:underline"
          >
            {t("viewAll")}
            <ArrowUpRight size={16} />
          </Link>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {featuredProjects.map((project) => (
            <ProjectCard
              key={project.slug}
              project={project}
              locale={locale}
              typeLabel={
                project.type === "builtin"
                  ? tProjects("type.builtin")
                  : tProjects("type.standalone")
              }
              openLabel={tProjects("open")}
              tryLabel={tProjects("tryNow")}
            />
          ))}
        </div>

        <div className="md:hidden mt-8 text-center">
          <Link
            href="/projects"
            className="inline-flex items-center gap-1 text-sm text-accent hover:underline"
          >
            {t("viewAll")}
            <ArrowUpRight size={16} />
          </Link>
        </div>
      </div>
    </section>
  );
}

function ProjectCard({
  project,
  locale,
  typeLabel,
  openLabel,
  tryLabel,
}: {
  project: Project;
  locale: "en" | "zh";
  typeLabel: string;
  openLabel: string;
  tryLabel: string;
}) {
  return (
    <Link
      href={`/projects/${project.slug}`}
      className="group gradient-border rounded-2xl p-6 card-hover"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="w-12 h-12 rounded-xl glass flex items-center justify-center">
          {project.type === "builtin" ? (
            <Wrench className="text-accent" size={22} />
          ) : (
            <ExternalLink className="text-accent-violet" size={22} />
          )}
        </div>
        <span className="text-xs px-2 py-1 rounded-full glass text-muted uppercase tracking-wide">
          {typeLabel}
        </span>
      </div>

      <h3 className="text-xl font-bold mb-2 group-hover:text-accent transition-colors">
        {project.title[locale]}
      </h3>
      <p className="text-sm text-secondary mb-4 line-clamp-2">
        {project.description[locale]}
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {project.tags.map((tag) => (
          <span
            key={tag}
            className="text-xs px-2 py-0.5 rounded-md glass text-muted"
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="flex items-center gap-1 text-sm text-accent">
        {project.type === "builtin" ? openLabel : tryLabel}
        <ArrowUpRight size={14} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
      </div>
    </Link>
  );
}
