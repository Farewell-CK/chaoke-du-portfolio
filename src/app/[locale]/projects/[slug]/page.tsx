import { setRequestLocale, getTranslations } from "next-intl/server";
import { notFound } from "next/navigation";
import { getAllProjects, getProjectBySlug } from "@/lib/projects";
import { Link } from "@/i18n/navigation";
import { ArrowLeft, ExternalLink, Wrench, ArrowUpRight } from "lucide-react";

export function generateStaticParams() {
  return getAllProjects().map((project) => ({ slug: project.slug }));
}

export default async function ProjectDetailPage({
  params,
}: {
  params: Promise<{ locale: string; slug: string }>;
}) {
  const { locale, slug } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("Projects");
  const tCommon = await getTranslations("Common");

  const project = getProjectBySlug(slug);
  if (!project) {
    notFound();
  }

  const loc = locale as "en" | "zh";

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-4xl mx-auto">
        <Link
          href="/projects"
          className="inline-flex items-center gap-1 text-sm text-muted hover:text-accent transition-colors mb-8"
        >
          <ArrowLeft size={16} />
          {tCommon("back")}
        </Link>

        <div className="flex items-start gap-4 mb-8">
          <div className="w-16 h-16 rounded-2xl glass flex items-center justify-center shrink-0">
            {project.type === "builtin" ? (
              <Wrench className="text-accent" size={28} />
            ) : (
              <ExternalLink className="text-accent-violet" size={28} />
            )}
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold mb-2">
              {project.title[loc]}
            </h1>
            <span className="text-xs px-3 py-1 rounded-full glass text-muted uppercase tracking-wide">
              {project.type === "builtin" ? t("type.builtin") : t("type.standalone")}
            </span>
          </div>
        </div>

        <p className="text-lg text-secondary mb-8">
          {project.description[loc]}
        </p>

        {project.longDescription && (
          <div className="prose prose-invert max-w-none mb-8">
            <p className="text-secondary leading-relaxed">
              {project.longDescription[loc]}
            </p>
          </div>
        )}

        <div className="flex flex-wrap gap-2 mb-8">
          {project.tags.map((tag) => (
            <span
              key={tag}
              className="text-sm px-3 py-1 rounded-lg glass text-secondary"
            >
              {tag}
            </span>
          ))}
        </div>

        <div className="flex gap-4">
          {project.url && (
            <a
              href={project.url}
              target={project.type === "standalone" ? "_blank" : undefined}
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-full text-white font-medium transition-all hover:scale-105"
              style={{ background: "var(--gradient-primary)" }}
            >
              {project.type === "builtin" ? t("open") : t("tryNow")}
              <ArrowUpRight size={18} />
            </a>
          )}
          {project.github && (
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-full glass text-secondary font-medium card-hover"
            >
              GitHub
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
