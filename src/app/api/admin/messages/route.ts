import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { verifyAuth, verifyApiKey } from "@/lib/auth";

export async function GET(request: Request) {
  if (!verifyAuth(request) && !verifyApiKey(request)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const messages = db.getAllMessages();
  return NextResponse.json(messages);
}
