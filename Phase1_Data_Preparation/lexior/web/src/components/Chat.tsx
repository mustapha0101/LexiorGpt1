import { useEffect, useRef } from "react";
import type { UseChatReturn } from "../hooks/useChat";
import { AgentProgress } from "./AgentProgress";
import { MessageBubble } from "./MessageBubble";
import { InputBar } from "./InputBar";

interface Props {
  chat: UseChatReturn;
}

export function Chat({ chat }: Props) {
  const {
    messages,
    streaming,
    currentNode,
    visitedNodes,
    sendMessage,
    cancelStream,
    clearMessages,
  } = chat;
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentNode]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between h-16 px-6 border-b border-border bg-surface shrink-0">
        <div>
          <h1 className="text-base font-semibold text-text-primary">
            Legal Assistant
          </h1>
          <p className="text-xs text-text-muted">
            Quebec &amp; Federal Law &middot; CCQ, CPC, Jurisprudence
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearMessages}
            className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-3 py-1.5 rounded-lg hover:bg-surface-raised"
          >
            Clear chat
          </button>
        )}
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <EmptyState onSuggestion={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Agent progress stepper */}
            {(streaming || visitedNodes.length > 0) && (
              <AgentProgress
                currentNode={currentNode}
                visitedNodes={visitedNodes}
              />
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <InputBar
        onSend={sendMessage}
        disabled={false}
        streaming={streaming}
        onCancel={cancelStream}
      />
    </div>
  );
}

/* ── Empty state with suggested queries ── */

const SUGGESTIONS = [
  "What are the conditions for resolving a lease in Quebec?",
  "Explain the regime of civil liability under art. 1457 CCQ.",
  "What is the prescription period for contractual claims?",
  "How does hypothecary recourse work for creditors?",
];

function EmptyState({ onSuggestion }: { onSuggestion: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-brand-100 flex items-center justify-center mb-5">
        <svg
          className="w-7 h-7 text-brand-600"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"
          />
        </svg>
      </div>

      <h2 className="text-lg font-semibold text-text-primary mb-1">
        Ask Lexior
      </h2>
      <p className="text-sm text-text-secondary mb-8 max-w-md">
        Get answers grounded in Quebec civil law and Canadian federal law.
        Lexior searches the CCQ, CPC, regulations, and case law to build
        precise legal analysis.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSuggestion(s)}
            className="
              text-left text-sm px-4 py-3 rounded-xl border border-border
              text-text-secondary hover:text-text-primary hover:bg-surface-raised
              hover:border-brand-300 transition-colors cursor-pointer
            "
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
