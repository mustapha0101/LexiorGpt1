/* ── SSE event types from POST /api/chat ── */

export interface ThinkingEvent {
  type: "thinking";
  content: string;
}

export interface RequestStartedEvent {
  type: "request_started";
  model: string;
  provider_model: string;
  /** Modèle qui JUGE la réponse — distinct du rédacteur en principe. */
  critic_model?: string;
  thread_id: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  /** Rang de l'observation, partagé avec le `tool_result` correspondant. */
  index: number;
  tool: string;
  args: Record<string, unknown>;
  /** Champs ignorés car absents du schéma courant de l'outil. */
  schema_correction?: string[];
}

export interface ToolResultEvent {
  type: "tool_result";
  /** Rang de l'observation, identique à celui du `tool_call`. */
  index: number;
  tool: string;
  result: string;
  ok: boolean;
  /**
   * Vrai quand l'observation a été RÉVISÉE après une première diffusion
   * (la vérification peut invalider un résultat d'abord annoncé valide).
   * Le couple `index` permet de remplacer la version affichée.
   */
  revised?: boolean;
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
  /**
   * Verdict RÉEL du contrôle d'acceptation. Faux pour une clarification en
   * attente, une réponse vide, un rejet ou une réponse de repli : ce champ
   * ne dit pas « le tour s'est terminé » mais « la réponse est fondée ».
   */
  accepted: boolean;
  /** Le graphe est suspendu sur une clarification (interrupt LangGraph) */
  pending_clarification?: boolean;
  /** Motif d'arrêt quand `accepted` est faux. */
  stop_reason?: string;
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
  | ErrorEvent;

/* ── Chat message model ── */

export type MessageRole = "user" | "assistant" | "tool" | "clarification";

export interface ToolCall {
  /** Rang de l'observation; sert à remplacer une version révisée. */
  index?: number;
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  ok?: boolean;
  classification?: string;
  reason?: string;
  schemaCorrection?: string[];
  metadata?: ToolResultMetadata;
  /** L'observation affichée a été révisée par la vérification. */
  revised?: boolean;
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

/* ── Chat model selection ── */

export type ChatModelId = "gpt-4o" | "gpt-4o-mini" | "qwen-local";

export const CHAT_MODEL_OPTIONS: { id: ChatModelId; label: string }[] = [
  { id: "gpt-4o", label: "GPT-4o" },
  { id: "gpt-4o-mini", label: "GPT-4o mini" },
  { id: "qwen-local", label: "Qwen 2.5 7B · local" },
];
