import type { ChatMessage } from "../types";
import { ToolCallCard } from "./ToolCallCard";

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  /* ── User message ── */
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl rounded-br-md px-4 py-3 bg-brand-600 text-white">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  /* ── Assistant message ── */
  return (
    <div className="flex justify-start">
      <div className="max-w-2xl w-full">
        {/* Thinking indicator */}
        {message.thinking && (
          <details className="mb-2 group">
            <summary className="text-xs text-text-muted cursor-pointer hover:text-text-secondary select-none">
              <span className="ml-1">Thinking...</span>
            </summary>
            <div className="mt-1 pl-4 border-l-2 border-border">
              <p className="text-xs text-text-muted whitespace-pre-wrap leading-relaxed">
                {message.thinking}
              </p>
            </div>
          </details>
        )}

        {/* Tool calls */}
        {message.toolCalls?.map((call, i) => (
          <ToolCallCard key={`${call.tool}-${i}`} call={call} />
        ))}

        {/* Content */}
        {message.content && (
          <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-surface-alt border border-border">
            <p className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">
              {message.content}
            </p>
          </div>
        )}

        {/* Agent status indicator */}
        {message.statusLabel && (
          <div className="flex items-center gap-2 px-3 py-1.5 mb-1">
            <span className="dot-pulse flex gap-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
            </span>
            <span className="text-xs text-text-muted">{message.statusLabel}</span>
          </div>
        )}

        {/* Streaming dots when content is empty but we are streaming */}
        {!message.content &&
          !message.thinking &&
          !message.statusLabel &&
          (!message.toolCalls || message.toolCalls.length === 0) && (
            <div className="rounded-2xl rounded-bl-md px-4 py-3 bg-surface-alt border border-border inline-block">
              <span className="dot-pulse flex gap-1">
                <span className="w-2 h-2 rounded-full bg-text-muted inline-block" />
                <span className="w-2 h-2 rounded-full bg-text-muted inline-block" />
                <span className="w-2 h-2 rounded-full bg-text-muted inline-block" />
              </span>
            </div>
          )}
      </div>
    </div>
  );
}
