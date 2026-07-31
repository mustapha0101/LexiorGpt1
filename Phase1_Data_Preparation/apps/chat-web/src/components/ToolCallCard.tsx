import { useState } from "react";
import type { ToolCall } from "../types";

interface Props {
  call: ToolCall;
}

/** Human-readable label for a tool name */
function formatToolName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function ToolCallCard({ call }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pending = call.result === undefined;

  return (
    <div
      className={`
        rounded-lg border text-sm my-2 overflow-hidden transition-colors
        ${pending ? "border-brand-300 bg-brand-50/50" : call.ok ? "border-border bg-surface-alt" : "border-error/30 bg-error/5"}
      `}
    >
      <button
        onClick={() => setExpanded((p) => !p)}
        className="w-full flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-surface-raised/50 transition-colors"
      >
        {/* Status indicator */}
        {pending ? (
          <span className="dot-pulse flex gap-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 inline-block" />
          </span>
        ) : call.ok ? (
          <svg className="w-4 h-4 text-success shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
          </svg>
        ) : (
          <svg className="w-4 h-4 text-error shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        )}

        <span className="font-medium text-text-primary truncate">
          {formatToolName(call.tool)}
        </span>

        {/* Classification du résultat : usable / candidate / irrelevant / empty… */}
        {call.classification && (
          <span
            title={call.reason || undefined}
            className={`text-[11px] px-1.5 py-0.5 rounded shrink-0 ${
              call.classification === "usable"
                ? "bg-success/10 text-success"
                : call.classification === "empty"
                  ? "bg-surface-raised text-text-muted"
                  : "bg-warning/10 text-warning"
            }`}
          >
            {call.classification}
          </span>
        )}

        {/* Chevron */}
        <svg
          className={`w-4 h-4 ml-auto text-text-muted transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {expanded && (
        <div className="border-t border-border px-3 py-2 space-y-2">
          {/* Arguments */}
          <div>
            <p className="text-xs font-medium text-text-muted mb-1">Arguments</p>
            <pre className="text-xs bg-surface-raised rounded p-2 overflow-x-auto whitespace-pre-wrap break-words text-text-secondary">
              {JSON.stringify(call.args, null, 2)}
            </pre>
          </div>

          {call.schemaCorrection && call.schemaCorrection.length > 0 && (
            <p className="text-xs text-warning">
              Champs ignorés car absents du schéma de l’outil : {call.schemaCorrection.join(", ")}
            </p>
          )}

          {/* Result */}
          {call.result !== undefined && (
            <div>
              <p className="text-xs font-medium text-text-muted mb-1">Result</p>
              <pre className="text-xs bg-surface-raised rounded p-2 overflow-x-auto whitespace-pre-wrap break-words text-text-secondary max-h-48 overflow-y-auto">
                {call.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
