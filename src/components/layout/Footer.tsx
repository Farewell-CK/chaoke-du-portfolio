"use client";

import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { Mail } from "lucide-react";

function GithubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

export default function Footer() {
  const t = useTranslations("Footer");
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-border glass relative-z mt-auto">
      <div className="max-w-7xl mx-auto px-6 py-10">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="text-sm text-muted text-center md:text-left">
            <div className="mb-1">
              <span className="gradient-text font-bold">Chaoke Du</span>
              <span className="mx-2 opacity-30">|</span>
              <span>© {year} {t("rights")}</span>
            </div>
            <a
              href="https://beian.miit.gov.cn/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs hover:text-accent transition-colors"
            >
              {t("icp")}
            </a>
          </div>

          <div className="text-xs text-muted/60">
            {t("builtWith")}
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://github.com/Farewell-CK"
              target="_blank"
              rel="noopener noreferrer"
              className="w-9 h-9 rounded-lg glass flex items-center justify-center text-muted hover:text-accent hover:border-accent/30 transition-all"
              aria-label="GitHub"
            >
              <GithubIcon size={18} />
            </a>
            <a
              href="mailto:contact@example.com"
              className="w-9 h-9 rounded-lg glass flex items-center justify-center text-muted hover:text-accent hover:border-accent/30 transition-all"
              aria-label="Email"
            >
              <Mail size={18} />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
