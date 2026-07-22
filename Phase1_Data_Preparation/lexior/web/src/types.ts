/* ── SSE event types from POST /api/chat ── */

export interface ThinkingEvent {
  type: "thinking";
  content: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  tool: string;
  args: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: "tool_result";
  tool: string;
  result: string;
  ok: boolean;
}

export interface TokenEvent {
  type: "token";
  content: string;
}

export interface ClarificationEvent {
  type: "clarification";
  question: string;
}

export interface StatusEvent {
  type: "status";
  node: string;
  label: string;
}

export interface DoneEvent {
  type: "done";
  accepted: boolean;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type SSEEvent =
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | TokenEvent
  | ClarificationEvent
  | StatusEvent
  | DoneEvent
  | ErrorEvent;

/* ── Chat message model ── */

export type MessageRole = "user" | "assistant" | "tool" | "clarification";

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  ok?: boolean;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  thinking?: string;
  toolCalls?: ToolCall[];
  /** For clarification messages */
  question?: string;
  /** Current agent status label */
  statusLabel?: string;
  timestamp: number;
}

/* ── Dataset dashboard types ── */

export interface DatasetRun {
  run_id: string;
  created_at: string;
  accepted: number;
  rejected: number;
  total: number;
  acceptance_rate: number;
}

export interface Rejection {
  scenario_id: string;
  reason: string;
  category?: string;
  details?: string;
}

/* ── Agent log entry ── */

export interface AgentLogEntry {
  id: string;
  node: string;
  label: string;
  timestamp: number;
  query: string;
}

/* ── App view ── */

export type AppView = "chat" | "dashboard";
