import { useCallback, useRef, useState } from "react";
import type {
  AgentLogEntry,
  ChatMessage,
  ChatModelId,
  RawSSELine,
  SSEEvent,
  ToolCall,
} from "../types";

let nextId = 0;
function uid(): string {
  return `msg_${Date.now()}_${nextId++}`;
}

function newThreadId(): string {
  return `live-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  streaming: boolean;
  currentNode: string | null;
  visitedNodes: string[];
  agentLog: AgentLogEntry[];
  rawEvents: RawSSELine[];
  model: ChatModelId;
  setModel: (model: ChatModelId) => void;
  sendMessage: (query: string) => Promise<void>;
  cancelStream: () => void;
  clearMessages: () => void;
}

const RAW_EVENTS_MAX = 500;

const MODEL_STORAGE_KEY = "lexior-chat-model";

function loadSavedModel(): ChatModelId {
  const saved = localStorage.getItem(MODEL_STORAGE_KEY);
  if (saved === "gpt-4o" || saved === "gpt-4o-mini" || saved === "qwen-local") {
    return saved;
  }
  return "gpt-4o";
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [visitedNodes, setVisitedNodes] = useState<string[]>([]);
  const [agentLog, setAgentLog] = useState<AgentLogEntry[]>([]);
  const [rawEvents, setRawEvents] = useState<RawSSELine[]>([]);
  const [model, setModelState] = useState<ChatModelId>(loadSavedModel);
  const abortRef = useRef<AbortController | null>(null);
  const queryRef = useRef("");
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;
  const modelRef = useRef(model);
  modelRef.current = model;
  /* Thread LangGraph de la conversation : permet au backend de
     reprendre une clarification en attente (interrupt/resume). */
  const threadIdRef = useRef<string>(newThreadId());

  const setModel = useCallback((next: ChatModelId) => {
    setModelState(next);
    localStorage.setItem(MODEL_STORAGE_KEY, next);
  }, []);

  const upsertAssistant = useCallback(
    (
      updater: (prev: ChatMessage) => ChatMessage,
      createIfMissing?: () => ChatMessage,
    ) => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant") {
          return [...prev.slice(0, -1), updater(last)];
        }
        if (createIfMissing) {
          return [...prev, createIfMissing()];
        }
        return prev;
      });
    },
    [],
  );

  const sendMessage = useCallback(
    async (query: string) => {
      abortRef.current?.abort();
      queryRef.current = query;

      const history = messagesRef.current
        .map((m) => ({
          role: m.role === "user" ? "user" : "assistant",
          content:
            m.role === "clarification" ? (m.question ?? "") : m.content,
        }))
        .filter((t) => t.content.trim().length > 0);

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: query,
        timestamp: Date.now(),
      };

      const assistantId = uid();
      const blankAssistant = (): ChatMessage => ({
        id: assistantId,
        role: "assistant",
        content: "",
        toolCalls: [],
        timestamp: Date.now(),
      });

      setMessages((prev) => [...prev, userMsg, blankAssistant()]);
      setStreaming(true);
      setCurrentNode(null);
      setVisitedNodes([]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            mode: "live",
            thread_id: threadIdRef.current,
            history,
            model: modelRef.current,
          }),
          signal: controller.signal,
        });

        if (!resp.ok || !resp.body) {
          throw new Error(`HTTP ${resp.status}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            const payload = trimmed.startsWith("data: ")
              ? trimmed.slice(6)
              : trimmed;

            let event: SSEEvent;
            try {
              event = JSON.parse(payload) as SSEEvent;
            } catch {
              continue;
            }

            setRawEvents((prev) => {
              const next = [
                ...prev,
                { id: uid(), eventType: event.type, line: trimmed },
              ];
              return next.length > RAW_EVENTS_MAX
                ? next.slice(next.length - RAW_EVENTS_MAX)
                : next;
            });

            switch (event.type) {
              case "thinking":
                upsertAssistant(
                  (m) => ({
                    ...m,
                    thinking: (m.thinking ?? "") + event.content,
                  }),
                  blankAssistant,
                );
                break;

              case "token":
                upsertAssistant(
                  (m) => ({ ...m, content: m.content + event.content }),
                  blankAssistant,
                );
                break;

              case "tool_call":
                upsertAssistant(
                  (m) => ({
                    ...m,
                    toolCalls: [
                      ...(m.toolCalls ?? []),
                      { tool: event.tool, args: event.args } as ToolCall,
                    ],
                  }),
                  blankAssistant,
                );
                break;

              case "tool_result":
                upsertAssistant((m) => {
                  const calls = [...(m.toolCalls ?? [])];
                  for (let i = calls.length - 1; i >= 0; i--) {
                    if (
                      calls[i]!.tool === event.tool &&
                      calls[i]!.result === undefined
                    ) {
                      calls[i] = {
                        ...calls[i]!,
                        result: event.result,
                        ok: event.ok,
                      };
                      break;
                    }
                  }
                  return { ...m, toolCalls: calls };
                }, blankAssistant);
                break;

              case "clarification":
                upsertAssistant(
                  (m) => ({
                    ...m,
                    content: event.question,
                    statusLabel: undefined,
                  }),
                  blankAssistant,
                );
                break;

              case "decision":
                setAgentLog((prev) => [
                  {
                    id: uid(),
                    node: "decision",
                    label: event.tool
                      ? `${event.decision} → ${event.tool}`
                      : event.decision,
                    timestamp: Date.now(),
                    query: queryRef.current.slice(0, 60),
                    step: event.step,
                    decision: event.decision,
                    tool: event.tool,
                    args: event.args,
                    jurisdiction: event.jurisdiction,
                    thinking: event.thinking,
                  },
                  ...prev,
                ].slice(0, 50));
                break;

              case "status":
                setCurrentNode(event.node);
                setVisitedNodes((prev) =>
                  prev.includes(event.node) ? prev : [...prev, event.node],
                );
                setAgentLog((prev) => [
                  {
                    id: uid(),
                    node: event.node,
                    label: event.label,
                    timestamp: Date.now(),
                    query: queryRef.current.slice(0, 60),
                  },
                  ...prev,
                ].slice(0, 50));
                upsertAssistant(
                  (m) => ({ ...m, statusLabel: event.label }),
                  blankAssistant,
                );
                break;

              case "error":
                setCurrentNode(null);
                upsertAssistant(
                  (m) => ({
                    ...m,
                    content:
                      m.content +
                      `\n\n**Error:** ${event.message}. Please try again.`,
                    statusLabel: undefined,
                  }),
                  blankAssistant,
                );
                break;

              case "done":
                setCurrentNode(null);
                upsertAssistant(
                  (m) => ({ ...m, statusLabel: undefined }),
                  blankAssistant,
                );
                break;
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          /* User-initiated cancel */
        } else {
          const errorContent =
            err instanceof Error ? err.message : "Connection lost";
          upsertAssistant(
            (m) => ({
              ...m,
              content:
                m.content +
                `\n\n**Error:** ${errorContent}. Please try again.`,
            }),
            blankAssistant,
          );
        }
      } finally {
        setStreaming(false);
        setCurrentNode(null);
        abortRef.current = null;
      }
    },
    [upsertAssistant],
  );

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clearMessages = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentNode(null);
    setVisitedNodes([]);
    setRawEvents([]);
    threadIdRef.current = newThreadId();
  }, []);

  return {
    messages,
    streaming,
    currentNode,
    visitedNodes,
    agentLog,
    rawEvents,
    model,
    setModel,
    sendMessage,
    cancelStream,
    clearMessages,
  };
}
