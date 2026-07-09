import { setRequestLocale } from "next-intl/server";
import Hero from "@/components/home/Hero";
import FeaturedProjects from "@/components/home/FeaturedProjects";
import LatestPosts from "@/components/home/LatestPosts";
import { getLatestPosts } from "@/lib/posts";

export default async function HomePage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);

  const latestPosts = getLatestPosts(3);

  return (
    <>
      <Hero />
      <FeaturedProjects />
      <LatestPosts posts={latestPosts} />
    </>
  );
}
