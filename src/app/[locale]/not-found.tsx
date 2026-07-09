import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center px-6 relative-z">
      <div className="text-center">
        <h1 className="text-8xl font-bold gradient-text mb-4">404</h1>
        <p className="text-secondary mb-8">Page not found</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass card-hover text-accent"
        >
          Back to Home
        </Link>
      </div>
    </div>
  );
}
