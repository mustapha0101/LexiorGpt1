import { useState } from "react";
import type { AppView } from "./types";
import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";
import { Dashboard } from "./components/Dashboard";

export default function App() {
  const [view, setView] = useState<AppView>("chat");
  const chat = useChat();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar
        currentView={view}
        onNavigate={setView}
        agentLog={chat.agentLog}
        rawEvents={chat.rawEvents}
        streaming={chat.streaming}
      />

      <main className="flex-1 min-w-0">
        {view === "chat" ? <Chat chat={chat} /> : <Dashboard />}
      </main>
    </div>
  );
}
