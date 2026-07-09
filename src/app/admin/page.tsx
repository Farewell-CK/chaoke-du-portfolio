"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, User } from "lucide-react";

export default function AdminLoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        setError("Invalid credentials");
        return;
      }

      const { token } = await res.json();
      localStorage.setItem("admin_token", token);
      router.push("/admin/dashboard");
    } catch {
      setError("Login failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-6 relative-z">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl glass flex items-center justify-center mx-auto mb-4 animate-glow-pulse">
            <Lock className="text-accent" size={28} />
          </div>
          <h1 className="text-2xl font-bold gradient-text">Admin Login</h1>
        </div>

        <form
          onSubmit={handleSubmit}
          className="glass rounded-2xl p-8 gradient-border space-y-4"
        >
          {error && (
            <div className="p-3 rounded-lg glass text-red-400 text-sm text-center">
              {error}
            </div>
          )}
          <div>
            <label className="block text-sm text-muted mb-2">Username</label>
            <div className="relative">
              <User
                size={16}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-muted"
              />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full pl-10 pr-4 py-3 rounded-lg glass border border-border text-foreground focus:outline-none focus:border-accent transition-colors"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm text-muted mb-2">Password</label>
            <div className="relative">
              <Lock
                size={16}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-muted"
              />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-3 rounded-lg glass border border-border text-foreground focus:outline-none focus:border-accent transition-colors"
              />
            </div>
          </div>
          <button
            type="submit"
            className="w-full py-3 rounded-full text-white font-medium transition-all hover:scale-[1.02]"
            style={{ background: "var(--gradient-primary)" }}
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
}
