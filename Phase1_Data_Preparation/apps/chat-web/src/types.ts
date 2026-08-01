/* ── SSE event types from POST /api/chat ── */

export interface ThinkingEvent {
  type: "thinking";
  content: string;
}

export interface RequestStartedEvent {
  type: "request_started";
  model: string;
  provider_model: string;
  thread_id: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  tool: string;
  args: Record<string, unknown>;
  /** Champs ignorés car absents du schéma courant de l'outil. */
  schema_correction?: string[];
}

export interface ToolResultEvent {
  type: "tool_result";
  tool: string;
  result: string;
  ok: boolean;
  /** usable | citable | candidate | irrelevant | empty | wrong_document_type… */
  classification?: string;
  /** Motif du classement, quand il y en a un. */
  reason?: string;
  metadata?: ToolResultMetadata;
  preview_truncated?: boolean;
  preview_character_count?: number;
}

export interface ToolResultMetadata {
  candidate_count?: number;
  candidate_articles?: string[];
  article_count?: number;
  article_numbers?: string[];
  preview_truncated?: boolean;
  preview_character_count?: number;
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

export interface DecisionEvent {
  type: "decision";
  step: number;
  decision: string;
  tool?: string | null;
  args?: Record<string, unknown>;
  jurisdiction?: string;
  thinking?: string;
}

export interface DoneEvent {
  type: "done";
  accepted: boolean;
  /** Le graphe est suspendu sur une clarification (interrupt LangGraph) */
  pending_clarification?: boolean;
  /** Thread LangGraph de la conversation */
  thread_id?: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export interface ObservabilityEvent {
  type: "observability";
  event: {
    node: string;
    task_id: string;
    thread_id: string;
    source_ids: string[];
    reason: string;
    status: string;
    timestamp: string;
  };
}

export interface EvaluationSaveErrorEvent {
  type: "evaluation_save_error";
  message: string;
}

export type SSEEvent =
  | RequestStartedEvent
  | ThinkingEvent
  | ToolCallEvent
  | ToolResultEvent
  | TokenEvent
  | ClarificationEvent
  | StatusEvent
  | ObservabilityEvent
  | DecisionEvent
  | DoneEvent
  | ErrorEvent
  | EvaluationSaveErrorEvent;

/* ── Chat message model ── */

export type MessageRole = "user" | "assistant" | "tool" | "clarification";

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  ok?: boolean;
  classification?: string;
  reason?: string;
  schemaCorrection?: string[];
  metadata?: ToolResultMetadata;
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

/* ── Raw SSE line (Agent Log raw view) ── */

export interface RawSSELine {
  id: string;
  /** Event type parsed from the line ("token", "tool_call", ...) */
  eventType: string;
  /** The line exactly as received on the wire, e.g. `data: {...}` */
  line: string;
}

/* ── Agent log entry ── */

export interface AgentLogEntry {
  id: string;
  node: string;
  label: string;
  timestamp: number;
  query: string;
  /** Planner-decision details (entries with node === "decision") */
  step?: number;
  decision?: string;
  tool?: string | null;
  args?: Record<string, unknown>;
  jurisdiction?: string;
  thinking?: string;
}

/* ── App view ── */

export type AppView = "chat" | "dashboard";

/* ── Chat model selection ── */

export type ChatModelId = "gpt-4o" | "gpt-4o-mini" | "qwen-local";

export type HumanEvaluationMode = "normal" | "human_40";

export interface HumanScenario {
  scenario_id: number;
  category: number;
  category_label: string;
  scenario_description: string;
  planned_order: number;
  actual_order: number | null;
  status: "not_started" | "in_progress" | "completed" | "interrupted";
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  thread_id: string | null;
  start_count: number;
  resume_count: number;
  human_query: string | null;
  expected_answer_note: string | null;
  human_evaluation: {
    overall_rating?: string;
    route_quality?: string;
    response_quality?: string[];
    notes?: string;
    observed_category?: string;
  } | null;
  final_result: { content?: string; accepted?: boolean; stop_reason?: string | null } | null;
  conversation: Array<Record<string, unknown>>;
  tool_calls: Array<Record<string, unknown>>;
  validation_events: Array<Record<string, unknown>>;
  timing: Record<string, unknown>;
}

export interface HumanEvaluationRun {
  schema_version: string;
  run: {
    run_id: string;
    started_at: string;
    completed_at: string | null;
    status: string;
    current_scenario_id: number | null;
    total_scenarios: number;
  };
  scenarios: HumanScenario[];
  run_summary: Record<string, unknown> | null;
}

export const CHAT_MODEL_OPTIONS: { id: ChatModelId; label: string }[] = [
  { id: "gpt-4o", label: "GPT-4o" },
  { id: "gpt-4o-mini", label: "GPT-4o mini" },
  { id: "qwen-local", label: "Qwen 2.5 7B · local" },
];
