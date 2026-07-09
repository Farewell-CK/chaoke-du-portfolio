import { NextResponse } from "next/server";
import { db } from "@/lib/db";

const MAX_NAME_LENGTH = 80;
const MAX_EMAIL_LENGTH = 254;
const MAX_CONTENT_LENGTH = 1000;

function isValidEmail(value: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

export async function GET() {
  const messages = db.getApprovedMessages();
  return NextResponse.json(messages);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const name = String(body.name || "Anonymous").trim() || "Anonymous";
    const email = body.email ? String(body.email).trim() : null;
    const content = String(body.content || "").trim();

    if (!content) {
      return NextResponse.json(
        { error: "Content is required" },
        { status: 400 }
      );
    }

    if (
      name.length > MAX_NAME_LENGTH ||
      content.length > MAX_CONTENT_LENGTH ||
      (email !== null && (email.length > MAX_EMAIL_LENGTH || !isValidEmail(email)))
    ) {
      return NextResponse.json({ error: "Invalid input" }, { status: 400 });
    }

    const id = db.addMessage(name, email, content);

    return NextResponse.json({ id, success: true });
  } catch {
    return NextResponse.json(
      { error: "Failed to post message" },
      { status: 500 }
    );
  }
}
