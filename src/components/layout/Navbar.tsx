"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { usePathname, useRouter } from "@/i18n/navigation";
import { Link } from "@/i18n/navigation";
import { Globe, Menu, X, Search } from "lucide-react";
import SearchModal from "./SearchModal";

export default function Navbar() {
  const t = useTranslations("Nav");
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navItems = [
    { href: "/", label: t("home") },
    { href: "/projects", label: t("projects") },
    { href: "/blog", label: t("blog") },
    { href: "/resources", label: t("resources") },
    { href: "/about", label: t("about") },
    { href: "/guestbook", label: t("guestbook") },
    { href: "/contact", label: t("contact") },
  ];

  const switchLocale = (locale: string) => {
    router.replace(pathname, { locale });
    setLangOpen(false);
  };

  const currentLocale = pathname.split("/")[0] || "en";

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "glass border-b"
          : "bg-transparent border-b border-transparent"
      }`}
    >
      <nav className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link
          href="/"
          className="text-lg font-bold tracking-tight"
        >
          <span className="gradient-text">CD</span>
        </Link>

        <div className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-2 text-sm transition-colors rounded-lg ${
                  isActive
                    ? "nav-active text-accent"
                    : "text-secondary hover:text-accent hover:bg-white/5"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setSearchOpen(true)}
            className="p-2 text-secondary hover:text-accent transition-colors rounded-lg hover:bg-white/5"
            aria-label="Search"
          >
            <Search size={18} />
          </button>
          <SearchModal isOpen={searchOpen} onClose={() => setSearchOpen(false)} />
          <div className="relative">
            <button
              onClick={() => setLangOpen(!langOpen)}
              className="p-2 text-secondary hover:text-accent transition-colors rounded-lg hover:bg-white/5 flex items-center gap-1"
              aria-label="Switch language"
            >
              <Globe size={18} />
              <span className="text-xs uppercase">{currentLocale}</span>
            </button>
            {langOpen && (
              <div className="absolute right-0 top-full mt-2 glass rounded-lg overflow-hidden min-w-[100px]">
                <button
                  onClick={() => switchLocale("en")}
                  className="block w-full text-left px-4 py-2 text-sm hover:bg-white/5 transition-colors"
                >
                  English
                </button>
                <button
                  onClick={() => switchLocale("zh")}
                  className="block w-full text-left px-4 py-2 text-sm hover:bg-white/5 transition-colors"
                >
                  中文
                </button>
              </div>
            )}
          </div>

          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 text-secondary hover:text-accent transition-colors"
            aria-label="Menu"
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </nav>

      {mobileOpen && (
        <div className="md:hidden glass border-t">
          <div className="px-6 py-4 flex flex-col gap-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className="px-3 py-2 text-sm text-secondary hover:text-accent transition-colors rounded-lg hover:bg-white/5"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
