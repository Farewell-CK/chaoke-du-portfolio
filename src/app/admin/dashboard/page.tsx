"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  MessageSquare,
  CheckCircle,
  Trash2,
  LogOut,
  Mail,
  FileText,
  BarChart3,
} from "lucide-react";

interface Message {
  id: number;
  name: string;
  email: string | null;
  content: string;
  status: string;
  created_at: string;
}

interface Contact {
  id: number;
  name: string;
  email: string;
  message: string;
  created_at: string;
}

type Tab = "messages" | "contacts" | "stats";

export default function AdminDashboard() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("messages");
  const [messages, setMessages] = useState<Message[]>([]);
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loading, setLoading] = useState(true);

  const token =
    typeof window !== "undefined" ? localStorage.getItem("admin_token") : null;

  useEffect(() => {
    if (!token) {
      router.push("/admin");
      return;
    }
    fetchData();
  }, [token]);

  const fetchData = async () => {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
    };

    try {
      const [msgRes, contactRes] = await Promise.all([
        fetch("/api/admin/messages", { headers }),
        fetch("/api/admin/contacts", { headers }),
      ]);

      if (msgRes.ok) setMessages(await msgRes.json());
      if (contactRes.ok) setContacts(await contactRes.json());
    } finally {
      setLoading(false);
    }
  };

  const approveMessage = async (id: number) => {
    await fetch(`/api/messages/${id}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ status: "approved" }),
    });
    fetchData();
  };

  const deleteMessage = async (id: number) => {
    await fetch(`/api/messages/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    fetchData();
  };

  const logout = () => {
    localStorage.removeItem("admin_token");
    router.push("/admin");
  };

  const pendingCount = messages.filter((m) => m.status === "pending").length;

  return (
    <div className="min-h-screen pt-24 pb-20 px-6 relative-z">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold gradient-text">Admin Dashboard</h1>
          <button
            onClick={logout}
            className="flex items-center gap-2 px-4 py-2 rounded-lg glass text-secondary hover:text-red-400 transition-colors"
          >
            <LogOut size={16} />
            Logout
          </button>
        </div>

        <div className="flex gap-2 mb-8">
          <TabButton
            active={tab === "messages"}
            onClick={() => setTab("messages")}
            icon={MessageSquare}
            label="Messages"
            badge={pendingCount > 0 ? pendingCount : undefined}
          />
          <TabButton
            active={tab === "contacts"}
            onClick={() => setTab("contacts")}
            icon={Mail}
            label="Contacts"
            badge={contacts.length || undefined}
          />
          <TabButton
            active={tab === "stats"}
            onClick={() => setTab("stats")}
            icon={BarChart3}
            label="Stats"
          />
        </div>

        {loading ? (
          <p className="text-muted text-center py-20">Loading...</p>
        ) : tab === "messages" ? (
          <div className="space-y-4">
            {messages.length === 0 ? (
              <p className="text-muted text-center py-20">No messages yet.</p>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className="glass rounded-xl p-5 gradient-border flex items-start justify-between gap-4"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="font-medium text-accent">{msg.name}</span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          msg.status === "approved"
                            ? "bg-green-500/20 text-green-400"
                            : msg.status === "pending"
                            ? "bg-yellow-500/20 text-yellow-400"
                            : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {msg.status}
                      </span>
                      <span className="text-xs text-muted">
                        {new Date(msg.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-secondary text-sm">{msg.content}</p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    {msg.status !== "approved" && (
                      <button
                        onClick={() => approveMessage(msg.id)}
                        className="p-2 rounded-lg glass text-green-400 hover:bg-green-500/10 transition-colors"
                        title="Approve"
                      >
                        <CheckCircle size={18} />
                      </button>
                    )}
                    <button
                      onClick={() => deleteMessage(msg.id)}
                      className="p-2 rounded-lg glass text-red-400 hover:bg-red-500/10 transition-colors"
                      title="Delete"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        ) : tab === "contacts" ? (
          <div className="space-y-4">
            {contacts.length === 0 ? (
              <p className="text-muted text-center py-20">No contacts yet.</p>
            ) : (
              contacts.map((c) => (
                <div
                  key={c.id}
                  className="glass rounded-xl p-5 gradient-border"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-medium text-accent">{c.name}</span>
                    <span className="text-sm text-muted">{c.email}</span>
                    <span className="text-xs text-muted ml-auto">
                      {new Date(c.created_at).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-secondary text-sm">{c.message}</p>
                </div>
              ))
            )}
          </div>
        ) : (
          <div className="grid md:grid-cols-3 gap-4">
            <StatCard
              icon={MessageSquare}
              label="Total Messages"
              value={messages.length}
            />
            <StatCard
              icon={CheckCircle}
              label="Pending Approval"
              value={pendingCount}
            />
            <StatCard
              icon={Mail}
              label="Contact Forms"
              value={contacts.length}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
  badge,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof MessageSquare;
  label: string;
  badge?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
        active
          ? "glass border-accent text-accent"
          : "glass text-muted hover:text-secondary"
      }`}
    >
      <Icon size={16} />
      {label}
      {badge !== undefined && (
        <span className="text-xs px-1.5 py-0.5 rounded-full bg-accent/20 text-accent">
          {badge}
        </span>
      )}
    </button>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof MessageSquare;
  label: string;
  value: number;
}) {
  return (
    <div className="glass rounded-xl p-6 gradient-border text-center">
      <Icon className="text-accent mx-auto mb-3" size={28} />
      <p className="text-3xl font-bold mb-1">{value}</p>
      <p className="text-sm text-muted">{label}</p>
    </div>
  );
}
