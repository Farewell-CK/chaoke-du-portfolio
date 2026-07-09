"use client";

import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";
import { MessageSquare, Send } from "lucide-react";

interface Message {
  id: number;
  name: string;
  content: string;
  status: string;
  created_at: string;
}

export default function GuestbookPage() {
  const t = useTranslations("Messages");
  const [messages, setMessages] = useState<Message[]>([]);
  const [content, setContent] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/messages")
      .then((res) => res.json())
      .then((data) => {
        setMessages(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;

    try {
      await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name || "Anonymous", content }),
      });
      setContent("");
      setName("");
      const res = await fetch("/api/messages");
      const data = await res.json();
      setMessages(data);
    } catch {
      // ignore
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center gap-3 mb-2">
          <MessageSquare className="text-accent" size={32} />
          <h1 className="text-4xl md:text-5xl font-bold gradient-text">
            {t("title")}
          </h1>
        </div>
        <p className="text-muted text-lg mb-12">{t("subtitle")}</p>

        <form
          onSubmit={handleSubmit}
          className="glass rounded-2xl p-6 gradient-border mb-12 space-y-4"
        >
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t("anonymous")}
            className="w-full px-4 py-2.5 rounded-lg glass border border-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition-colors"
          />
          <textarea
            required
            rows={3}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={t("placeholder")}
            className="w-full px-4 py-2.5 rounded-lg glass border border-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition-colors resize-none"
          />
          <button
            type="submit"
            className="flex items-center gap-2 px-6 py-2.5 rounded-full text-white font-medium transition-all hover:scale-105"
            style={{ background: "var(--gradient-primary)" }}
          >
            <Send size={16} />
            {t("submit")}
          </button>
        </form>

        {loading ? (
          <p className="text-muted text-center">Loading...</p>
        ) : messages.length === 0 ? (
          <p className="text-muted text-center">No messages yet.</p>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className="glass rounded-xl p-5 gradient-border card-hover"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-accent">{msg.name}</span>
                  <span className="text-xs text-muted">
                    {new Date(msg.created_at).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-secondary text-sm">{msg.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
