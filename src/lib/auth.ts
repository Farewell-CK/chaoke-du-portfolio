import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";

const JWT_SECRET = process.env.ADMIN_JWT_SECRET || "change-me-in-production";
const ADMIN_USERNAME = process.env.ADMIN_USERNAME || "admin";
const ADMIN_PASSWORD_HASH = bcrypt.hashSync(
  process.env.ADMIN_PASSWORD || "admin123",
  10
);

export function verifyCredentials(username: string, password: string): boolean {
  return (
    username === ADMIN_USERNAME &&
    bcrypt.compareSync(password, ADMIN_PASSWORD_HASH)
  );
}

export function generateToken(): string {
  return jwt.sign({ role: "admin" }, JWT_SECRET, { expiresIn: "7d" });
}

export async function verifyAuth(request: Request): Promise<boolean> {
  const authHeader = request.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return false;
  }

  try {
    const token = authHeader.slice(7);
    jwt.verify(token, JWT_SECRET);
    return true;
  } catch {
    return false;
  }
}

export function getApiKey(): string {
  return process.env.PORTFOLIO_API_KEY || JWT_SECRET;
}

export function verifyApiKey(request: Request): boolean {
  const apiKey = request.headers.get("x-api-key");
  if (!apiKey) return false;
  return apiKey === getApiKey();
}
