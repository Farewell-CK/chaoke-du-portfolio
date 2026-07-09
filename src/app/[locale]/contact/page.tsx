"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";
import { Mail } from "lucide-react";

function GithubIcon({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

export default function ContactPage() {
  const t = useTranslations("Contact");
  const [sent, setSent] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", message: "" });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      setSent(true);
      setForm({ name: "", email: "", message: "" });
    } catch {
      setSent(true);
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl md:text-5xl font-bold mb-2 gradient-text">
          {t("title")}
        </h1>
        <p className="text-muted text-lg mb-12">{t("subtitle")}</p>

        <div className="grid md:grid-cols-2 gap-4 mb-12">
          <a
            href="mailto:contact@example.com"
            className="glass rounded-xl p-6 flex items-center gap-4 card-hover gradient-border"
          >
            <div className="w-12 h-12 rounded-xl glass flex items-center justify-center">
              <Mail className="text-accent" size={24} />
            </div>
            <div>
              <p className="text-sm text-muted">{t("email")}</p>
              <p className="font-medium">contact@example.com</p>
            </div>
          </a>

          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="glass rounded-xl p-6 flex items-center gap-4 card-hover gradient-border"
          >
            <div className="w-12 h-12 rounded-xl glass flex items-center justify-center">
              <GithubIcon size={24} />
            </div>
            <div>
              <p className="text-sm text-muted">{t("github")}</p>
              <p className="font-medium">@yourusername</p>
            </div>
          </a>
        </div>

        <form onSubmit={handleSubmit} className="glass rounded-2xl p-8 gradient-border space-y-4">
          {sent && (
            <div className="p-4 rounded-lg glass text-accent text-sm">
              {t("sent")}
            </div>
          )}
          <div>
            <label className="block text-sm text-muted mb-2">{t("name")}</label>
            <input
              type="text"
              required
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder={t("namePlaceholder")}
              className="w-full px-4 py-3 rounded-lg glass border border-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm text-muted mb-2">{t("email")}</label>
            <input
              type="email"
              required
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              placeholder="you@example.com"
              className="w-full px-4 py-3 rounded-lg glass border border-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm text-muted mb-2">{t("message")}</label>
            <textarea
              required
              rows={5}
              value={form.message}
              onChange={(e) => setForm({ ...form, message: e.target.value })}
              placeholder={t("messagePlaceholder")}
              className="w-full px-4 py-3 rounded-lg glass border border-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition-colors resize-none"
            />
          </div>
          <button
            type="submit"
            className="w-full py-3 rounded-full text-white font-medium transition-all hover:scale-[1.02]"
            style={{ background: "var(--gradient-primary)" }}
          >
            {t("send")}
          </button>
        </form>
      </div>
    </div>
  );
}
