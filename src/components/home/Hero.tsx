"use client";

import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { ArrowRight, BookOpen } from "lucide-react";

export default function Hero() {
  const t = useTranslations("Hero");

  return (
    <section className="min-h-[90vh] flex items-center justify-center px-6 pt-16 relative-z">
      <div className="max-w-4xl mx-auto text-center">
        <div className="animate-fade-in-up inline-block mb-6">
          <span className="text-xs uppercase tracking-widest text-accent glow-text-cyan px-4 py-1.5 rounded-full glass border border-accent/20">
            {t("greeting")} 👋
          </span>
        </div>

        <h1
          className="animate-fade-in-up text-5xl md:text-7xl font-bold mb-4 gradient-text animate-gradient"
          style={{ animationDelay: "0.1s" }}
        >
          {t("name")}
        </h1>

        <p
          className="animate-fade-in-up text-xl md:text-2xl text-secondary mb-2"
          style={{ animationDelay: "0.2s" }}
        >
          {t("role")}
        </p>

        <p
          className="animate-fade-in-up text-lg text-muted mb-10 max-w-2xl mx-auto"
          style={{ animationDelay: "0.3s" }}
        >
          {t("tagline")}
        </p>

        <div
          className="animate-fade-in-up flex flex-col sm:flex-row items-center justify-center gap-4"
          style={{ animationDelay: "0.4s" }}
        >
          <Link
            href="/projects"
            className="group flex items-center gap-2 px-8 py-3.5 rounded-full text-white font-medium transition-all hover:scale-105"
            style={{ background: "var(--gradient-primary)" }}
          >
            {t("cta")}
            <ArrowRight
              size={18}
              className="transition-transform group-hover:translate-x-1"
            />
          </Link>
          <Link
            href="/blog"
            className="group flex items-center gap-2 px-8 py-3.5 rounded-full text-secondary font-medium transition-all glass card-hover"
          >
            <BookOpen size={18} />
            {t("ctaSecondary")}
          </Link>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 hidden md:block">
        <div className="w-6 h-10 rounded-full border-2 border-border flex items-start justify-center p-1.5">
          <div className="w-1 h-2 rounded-full bg-accent animate-float" />
        </div>
      </div>
    </section>
  );
}
