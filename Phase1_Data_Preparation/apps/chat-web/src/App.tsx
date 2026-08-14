import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";

export default function App() {
  const chat = useChat();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar
        agentLog={chat.agentLog}
        rawEvents={chat.rawEvents}
        streaming={chat.streaming}
      />

      <main className="flex-1 min-w-0">
        <Chat chat={chat} />
      </main>
    </div>
  );
}
