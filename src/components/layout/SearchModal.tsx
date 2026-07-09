"use client";

import { useState, useEffect, useRef } from "react";
import { Search, X, FileText, Code2, Book } from "lucide-react";
import { useRouter } from "next/navigation";

interface SearchResult {
  type: string;
  title: string;
  description: string;
  url: string;
  tags: string[];
}

export default function SearchModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!query || query.length < 2) {
      setResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        setResults(data.results || []);
      } finally {
        setLoading(false);
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const typeIcons: Record<string, typeof FileText> = {
    project: Code2,
    blog: FileText,
    resource: Book,
  };

  const handleResultClick = (url: string) => {
    if (url.startsWith("http")) {
      window.open(url, "_blank");
    } else {
      router.push(url);
    }
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-20 px-4"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-2xl glass rounded-2xl gradient-border overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 p-4 border-b border-border">
          <Search size={20} className="text-muted shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search projects, blog posts, resources..."
            className="flex-1 bg-transparent text-foreground placeholder:text-muted focus:outline-none"
          />
          <button
            onClick={onClose}
            className="text-muted hover:text-foreground transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {loading && (
            <div className="p-8 text-center text-muted">Searching...</div>
          )}
          {!loading && query.length >= 2 && results.length === 0 && (
            <div className="p-8 text-center text-muted">No results found.</div>
          )}
          {!loading &&
            results.map((result, i) => {
              const Icon = typeIcons[result.type] || FileText;
              return (
                <button
                  key={i}
                  onClick={() => handleResultClick(result.url)}
                  className="w-full text-left p-4 hover:bg-white/5 transition-colors flex items-start gap-3 border-b border-border/50"
                >
                  <div className="w-10 h-10 rounded-lg glass flex items-center justify-center shrink-0">
                    <Icon className="text-accent" size={18} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">
                        {result.title}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full glass text-muted uppercase">
                        {result.type}
                      </span>
                    </div>
                    <p className="text-sm text-muted truncate">
                      {result.description}
                    </p>
                    {result.tags.length > 0 && (
                      <div className="flex gap-1 mt-1">
                        {result.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="text-xs text-muted/60"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </button>
              );
            })}
        </div>
      </div>
    </div>
  );
}
