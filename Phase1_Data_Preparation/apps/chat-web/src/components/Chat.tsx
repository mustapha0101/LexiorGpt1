import { useEffect, useRef, useState } from "react";
import type { UseChatReturn } from "../hooks/useChat";
import { CHAT_MODEL_OPTIONS, type ChatMessage, type ChatModelId } from "../types";
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
    rawEvents,
    model,
    setModel,
    sendMessage,
    cancelStream,
    clearMessages,
  } = chat;
  const bottomRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState<"conversation" | "raw" | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentNode]);

  async function copyText(kind: "conversation" | "raw") {
    const text = kind === "conversation"
      ? formatConversation(messages)
      : rawEvents.map((event) => event.line).join("\n");
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    setCopied(kind);
    window.setTimeout(() => setCopied((current) => current === kind ? null : current), 1600);
  }

  return (
    <div className="flex flex-col h-full min-h-0">
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
        <div className="flex items-center gap-2">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as ChatModelId)}
            title="Model used for planning and answers"
            className="
              text-xs bg-surface-raised border border-border rounded-lg
              px-2.5 py-1.5 text-text-secondary cursor-pointer
              focus:outline-none focus:ring-2 focus:ring-brand-500/40
            "
          >
            {CHAT_MODEL_OPTIONS.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-3 py-1.5 rounded-lg hover:bg-surface-raised"
            >
              Clear chat
            </button>
          )}
          <button
            onClick={() => copyText("conversation")}
            disabled={messages.length === 0}
            title="Copier la conversation avec le thinking public, les outils et la réponse"
            className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-2.5 py-1.5 rounded-lg hover:bg-surface-raised disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {copied === "conversation" ? "Copié" : "Copier conversation"}
          </button>
          <button
            onClick={() => copyText("raw")}
            disabled={rawEvents.length === 0}
            title="Copier les lignes raw exactes reçues du log SSE"
            className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-2.5 py-1.5 rounded-lg hover:bg-surface-raised disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {copied === "raw" ? "Copié" : "Copier raw"}
          </button>
        </div>
      </header>

      <div className="flex-1 min-h-0 flex flex-col">
        {/* Messages stay above the evaluation form so the final answer remains visible. */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
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

function formatConversation(messages: ChatMessage[]): string {
  return messages.map((message) => {
    const sections: string[] = [
      `===== ${message.role.toUpperCase()} =====`,
    ];
    if (message.thinking?.trim()) {
      sections.push(`[thinking public]\n${message.thinking.trim()}`);
    }
    if (message.toolCalls?.length) {
      sections.push(message.toolCalls.map((tool, index) => [
        `[outil ${index + 1}] ${tool.tool}`,
        `arguments: ${JSON.stringify(tool.args)}`,
        tool.result !== undefined ? `résultat: ${tool.result}` : "",
        tool.classification ? `classification: ${tool.classification}` : "",
      ].filter(Boolean).join("\n")).join("\n\n"));
    }
    if (message.content.trim()) sections.push(message.content.trim());
    if (message.statusLabel) sections.push(`[statut] ${message.statusLabel}`);
    return sections.join("\n\n");
  }).join("\n\n");
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
