import { useEffect, useRef, useState } from "react";
import type { AgentLogEntry, AppView, RawSSELine } from "../types";

interface Props {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
  agentLog: AgentLogEntry[];
  rawEvents: RawSSELine[];
  streaming: boolean;
}

/* Couleur par type d'événement dans la vue raw */
const RAW_COLORS: Record<string, string> = {
  tool_call: "text-amber-300",
  tool_result: "text-emerald-300",
  decision: "text-blue-300",
  clarification: "text-purple-300",
  error: "text-red-300",
};

interface RawDisplayItem {
  id: string;
  kind: "line" | "tokens";
  eventType?: string;
  text: string;
}

/** Collapse consecutive token events into a single counter line. */
function collapseRaw(events: RawSSELine[]): RawDisplayItem[] {
  const out: RawDisplayItem[] = [];
  let tokenRun = 0;
  let runId = "";
  for (const ev of events) {
    if (ev.eventType === "token") {
      if (tokenRun === 0) runId = ev.id;
      tokenRun += 1;
      continue;
    }
    if (tokenRun > 0) {
      out.push({
        id: runId,
        kind: "tokens",
        text: `… ${tokenRun} × {"type": "token"} …`,
      });
      tokenRun = 0;
    }
    out.push({ id: ev.id, kind: "line", eventType: ev.eventType, text: ev.line });
  }
  if (tokenRun > 0) {
    out.push({
      id: runId,
      kind: "tokens",
      text: `… ${tokenRun} × {"type": "token"} …`,
    });
  }
  return out;
}

function ChatIcon() {
  return (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 0 1 1.037-.443 48.3 48.3 0 0 0 5.862-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.4 48.4 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"
      />
    </svg>
  );
}

function DashboardIcon() {
  return (
    <svg
      className="w-5 h-5"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"
      />
    </svg>
  );
}

const NODE_COLORS: Record<string, string> = {
  initialize: "bg-slate-400",
  classify_request: "bg-sky-400",
  classify_follow_up: "bg-sky-400",
  update_active_task: "bg-sky-400",
  resolve_jurisdiction: "bg-indigo-400",
  analyze_facts: "bg-sky-400",
  plan: "bg-blue-400",
  validate_plan: "bg-blue-500",
  execute_tool: "bg-amber-400",
  verify_tool_result: "bg-amber-500",
  classify_tool_result: "bg-yellow-400",
  update_research_state: "bg-lime-400",
  reformulate_search: "bg-orange-300",
  build_answer_contract: "bg-emerald-300",
  handle_clarification: "bg-purple-400",
  generate_answer: "bg-emerald-400",
  run_critics: "bg-cyan-400",
  classify_failures: "bg-rose-300",
  repair_answer: "bg-orange-400",
  repair_trajectory: "bg-orange-500",
  validate_final: "bg-teal-400",
  compute_acceptance: "bg-teal-500",
  export_dataset: "bg-green-500",
  return_live_answer: "bg-green-500",
  reject: "bg-red-400",
};

const DECISION_COLORS: Record<string, string> = {
  call_tool: "bg-amber-400",
  final_answer: "bg-green-500",
  ask_clarification: "bg-purple-400",
  cannot_conclude: "bg-red-400",
};

function timeAgo(ts: number): string {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 5) return "now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

const navItems: { view: AppView; label: string; Icon: () => JSX.Element }[] = [
  { view: "chat", label: "Chat", Icon: ChatIcon },
  { view: "dashboard", label: "Dashboard", Icon: DashboardIcon },
];

export function Sidebar({
  currentView,
  onNavigate,
  agentLog,
  rawEvents,
  streaming,
}: Props) {
  const [logView, setLogView] = useState<"log" | "raw">("log");
  const rawBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logView === "raw") {
      rawBottomRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [logView, rawEvents]);

  return (
    <aside className="flex flex-col w-64 border-r border-border bg-surface-alt shrink-0">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-border">
        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
          <span className="text-white font-bold text-sm">L</span>
        </div>
        <span className="font-semibold text-lg tracking-tight text-text-primary">
          Lexior
        </span>
      </div>

      {/* Navigation */}
      <nav className="p-3 space-y-1">
        {navItems.map(({ view, label, Icon }) => {
          const active = currentView === view;
          return (
            <button
              key={view}
              onClick={() => onNavigate(view)}
              className={`
                w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-colors duration-150 cursor-pointer
                ${
                  active
                    ? "bg-brand-600 text-white"
                    : "text-text-secondary hover:bg-surface-raised hover:text-text-primary"
                }
              `}
            >
              <Icon />
              {label}
            </button>
          );
        })}
      </nav>

      {/* Agent Activity Log */}
      <div className="flex-1 min-h-0 flex flex-col border-t border-border">
        <div className="flex items-center gap-2 px-4 py-2.5">
          <span className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">
            Agent Log
          </span>
          {streaming && (
            <span className="flex gap-0.5">
              <span className="w-1 h-1 rounded-full bg-brand-500 animate-pulse" />
              <span className="w-1 h-1 rounded-full bg-brand-500 animate-pulse [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-brand-500 animate-pulse [animation-delay:300ms]" />
            </span>
          )}
          <span className="ml-auto flex rounded-md border border-border overflow-hidden">
            {(["log", "raw"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setLogView(mode)}
                className={`px-2 py-0.5 text-[10px] font-medium transition-colors cursor-pointer ${
                  logView === mode
                    ? "bg-brand-600 text-white"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {mode}
              </button>
            ))}
          </span>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3">
          {logView === "raw" ? (
            rawEvents.length === 0 ? (
              <p className="text-[11px] text-text-muted/60 px-1 py-2">
                No events yet
              </p>
            ) : (
              <div className="space-y-1 font-mono">
                {collapseRaw(rawEvents).map((item) =>
                  item.kind === "tokens" ? (
                    <p
                      key={item.id}
                      className="text-[9px] text-text-muted/50 select-none"
                    >
                      {item.text}
                    </p>
                  ) : (
                    <pre
                      key={item.id}
                      className={`text-[9.5px] leading-snug whitespace-pre-wrap break-all ${
                        RAW_COLORS[item.eventType ?? ""] ?? "text-text-muted"
                      }`}
                    >
                      {item.text}
                    </pre>
                  ),
                )}
                <div ref={rawBottomRef} />
              </div>
            )
          ) : agentLog.length === 0 ? (
            <p className="text-[11px] text-text-muted/60 px-1 py-2">
              No activity yet
            </p>
          ) : (
            <div className="space-y-0.5">
              {agentLog.map((entry) =>
                entry.node === "decision" ? (
                  <details
                    key={entry.id}
                    className="px-2 py-1.5 rounded-md hover:bg-surface-raised/50 transition-colors"
                  >
                    <summary className="flex items-start gap-2 cursor-pointer list-none">
                      <span
                        className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                          DECISION_COLORS[entry.decision ?? ""] ??
                          "bg-text-muted"
                        }`}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-[11px] font-medium text-text-primary truncate">
                          étape {entry.step} · {entry.label}
                        </p>
                        {entry.jurisdiction && (
                          <p className="text-[10px] text-text-muted truncate">
                            juridiction : {entry.jurisdiction}
                          </p>
                        )}
                      </div>
                      <span className="text-[10px] text-text-muted/60 whitespace-nowrap shrink-0 mt-0.5">
                        {timeAgo(entry.timestamp)}
                      </span>
                    </summary>
                    <div className="mt-1 ml-3.5 space-y-1">
                      {entry.args && Object.keys(entry.args).length > 0 && (
                        <pre className="text-[10px] text-text-secondary bg-surface rounded px-1.5 py-1 whitespace-pre-wrap break-all font-mono">
                          {JSON.stringify(entry.args, null, 1)}
                        </pre>
                      )}
                      {entry.thinking && (
                        <p className="text-[10px] text-text-muted italic leading-relaxed">
                          {entry.thinking}
                        </p>
                      )}
                    </div>
                  </details>
                ) : (
                  <div
                    key={entry.id}
                    className="flex items-start gap-2 px-2 py-1.5 rounded-md hover:bg-surface-raised/50 transition-colors"
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                        NODE_COLORS[entry.node] ?? "bg-text-muted"
                      }`}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] font-medium text-text-secondary truncate">
                        {entry.label}
                      </p>
                      <p className="text-[10px] text-text-muted truncate">
                        {entry.query}
                      </p>
                    </div>
                    <span className="text-[10px] text-text-muted/60 whitespace-nowrap shrink-0 mt-0.5">
                      {timeAgo(entry.timestamp)}
                    </span>
                  </div>
                ),
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <p className="text-xs text-text-muted">Lexior Legal AI</p>
        <p className="text-xs text-text-muted">Quebec &amp; Federal Law</p>
      </div>
    </aside>
  );
}
