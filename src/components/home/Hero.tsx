"use client";

import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { ArrowRight, BookOpen } from "lucide-react";

export default function Hero() {
  const t = useTranslations("Hero");

  return (
    <section className="min-h-[90vh] flex items-center justify-center px-6 pt-16 relative-z overflow-hidden">
      <div
        className="hero-orb"
        style={{
          width: 400,
          height: 400,
          background: "var(--accent-violet)",
          top: "10%",
          left: "5%",
        }}
      />
      <div
        className="hero-orb"
        style={{
          width: 350,
          height: 350,
          background: "var(--accent-cyan)",
          bottom: "5%",
          right: "8%",
          animationDelay: "3s",
        }}
      />

      <div className="max-w-4xl mx-auto text-center relative-z">
        <div className="animate-fade-in-up inline-block mb-8">
          <span className="text-xs uppercase tracking-[0.3em] text-accent glow-text-cyan px-5 py-2 rounded-full glass border border-accent/20 font-mono">
            {t("greeting")}
          </span>
        </div>

        <h1
          className="animate-fade-in-up text-6xl md:text-8xl font-bold mb-6 gradient-text animate-gradient tracking-tight"
          style={{ animationDelay: "0.1s" }}
        >
          {t("name")}
        </h1>

        <div
          className="animate-fade-in-up flex items-center justify-center gap-3 mb-4"
          style={{ animationDelay: "0.2s" }}
        >
          <span className="h-px w-12 bg-gradient-to-r from-transparent to-accent/50" />
          <p className="text-lg md:text-xl text-accent font-medium tracking-wide">
            {t("role")}
          </p>
          <span className="h-px w-12 bg-gradient-to-l from-transparent to-accent/50" />
        </div>

        <p
          className="animate-fade-in-up text-lg md:text-xl text-muted mb-12 max-w-2xl mx-auto leading-relaxed"
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
            className="group flex items-center gap-2 px-8 py-3.5 rounded-full text-white font-medium transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(0,212,255,0.4)]"
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
