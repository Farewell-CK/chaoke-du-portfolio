import fs from "fs";
import path from "path";

const dataDir = path.join(process.cwd(), "data");
const dbPath = path.join(dataDir, "store.json");

interface DBData {
  messages: Message[];
  contacts: Contact[];
  resources: DBResource[];
}

export interface Message {
  id: number;
  name: string;
  email: string | null;
  content: string;
  status: string;
  created_at: string;
}

export interface Contact {
  id: number;
  name: string;
  email: string;
  message: string;
  created_at: string;
}

interface DBResource {
  id: string;
  title_en: string;
  title_zh: string;
  description_en: string;
  description_zh: string;
  url: string;
  category: string;
  tags: string;
  created_at: string;
}

let data: DBData | null = null;

function loadData(): DBData {
  if (data) return data;

  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  if (fs.existsSync(dbPath)) {
    data = JSON.parse(fs.readFileSync(dbPath, "utf8"));
  } else {
    data = { messages: [], contacts: [], resources: [] };
    saveData();
  }

  return data!;
}

function saveData() {
  if (!data) return;
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  fs.writeFileSync(dbPath, JSON.stringify(data, null, 2));
}

function nextId(items: { id: number }[]): number {
  return items.length > 0 ? Math.max(...items.map((i) => i.id)) + 1 : 1;
}

function now(): string {
  return new Date().toISOString();
}

export const db = {
  addMessage(name: string, email: string | null, content: string): number {
    const d = loadData();
    const id = nextId(d.messages);
    d.messages.push({
      id,
      name: name || "Anonymous",
      email,
      content,
      status: "pending",
      created_at: now(),
    });
    saveData();
    return id;
  },

  getApprovedMessages(): Message[] {
    return loadData()
      .messages.filter((m) => m.status === "approved")
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  },

  getAllMessages(): Message[] {
    return loadData()
      .messages.slice()
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  },

  updateMessageStatus(id: number, status: string): void {
    const d = loadData();
    const msg = d.messages.find((m) => m.id === id);
    if (msg) {
      msg.status = status;
      saveData();
    }
  },

  deleteMessage(id: number): void {
    const d = loadData();
    d.messages = d.messages.filter((m) => m.id !== id);
    saveData();
  },

  addContact(name: string, email: string, message: string): void {
    const d = loadData();
    d.contacts.push({
      id: nextId(d.contacts),
      name,
      email,
      message,
      created_at: now(),
    });
    saveData();
  },

  getAllContacts(): Contact[] {
    return loadData()
      .contacts.slice()
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  },
};
