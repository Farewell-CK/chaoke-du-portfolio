"use client";

import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import type { Project } from "@/lib/projects";
import { ArrowUpRight, Wrench, ExternalLink } from "lucide-react";

export default function ProjectCard({
  project,
  locale,
}: {
  project: Project;
  locale: "en" | "zh";
}) {
  const t = useTranslations("Projects");

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
          {project.type === "builtin" ? t("type.builtin") : t("type.standalone")}
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
        {project.type === "builtin" ? t("open") : t("tryNow")}
        <ArrowUpRight
          size={14}
          className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
        />
      </div>
    </Link>
  );
}
