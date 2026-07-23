import { useCallback, useEffect, useState } from "react";
import type { DatasetRun, Rejection } from "../types";

export function Dashboard() {
  const [runs, setRuns] = useState<DatasetRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [rejections, setRejections] = useState<Record<string, Rejection[]>>({});
  const [loadingRejections, setLoadingRejections] = useState<string | null>(
    null,
  );

  /* Fetch runs on mount */
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const resp = await fetch("/api/dataset/runs");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: DatasetRun[] = await resp.json();
        if (!cancelled) {
          setRuns(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load runs");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  /* Fetch rejections for a run */
  const loadRejections = useCallback(
    async (runId: string) => {
      if (rejections[runId]) return; // already cached
      setLoadingRejections(runId);
      try {
        const resp = await fetch(`/api/dataset/runs/${runId}/rejections`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: Rejection[] = await resp.json();
        setRejections((prev) => ({ ...prev, [runId]: data }));
      } catch {
        /* silently ignore */
      } finally {
        setLoadingRejections(null);
      }
    },
    [rejections],
  );

  function toggleRun(runId: string) {
    if (expandedRun === runId) {
      setExpandedRun(null);
    } else {
      setExpandedRun(runId);
      void loadRejections(runId);
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between h-16 px-6 border-b border-border bg-surface shrink-0">
        <div>
          <h1 className="text-base font-semibold text-text-primary">
            Dataset Dashboard
          </h1>
          <p className="text-xs text-text-muted">
            Generation runs, acceptance rates, and rejection analysis
          </p>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-4xl mx-auto">
          {loading && (
            <div className="flex items-center justify-center py-20">
              <span className="dot-pulse flex gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-brand-500 inline-block" />
                <span className="w-2.5 h-2.5 rounded-full bg-brand-500 inline-block" />
                <span className="w-2.5 h-2.5 rounded-full bg-brand-500 inline-block" />
              </span>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-error/30 bg-error/5 px-5 py-4 text-sm text-error">
              {error}
            </div>
          )}

          {!loading && !error && runs.length === 0 && (
            <div className="text-center py-20">
              <p className="text-text-muted text-sm">No generation runs yet.</p>
            </div>
          )}

          {!loading && runs.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface-alt text-text-secondary">
                    <th className="text-left px-4 py-3 font-medium">Run ID</th>
                    <th className="text-left px-4 py-3 font-medium">Date</th>
                    <th className="text-right px-4 py-3 font-medium">
                      Accepted
                    </th>
                    <th className="text-right px-4 py-3 font-medium">
                      Rejected
                    </th>
                    <th className="text-right px-4 py-3 font-medium">Total</th>
                    <th className="text-right px-4 py-3 font-medium">Rate</th>
                    <th className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => {
                    const isExpanded = expandedRun === run.run_id;
                    return (
                      <RunRow
                        key={run.run_id}
                        run={run}
                        expanded={isExpanded}
                        onToggle={() => toggleRun(run.run_id)}
                        rejections={isExpanded ? rejections[run.run_id] : undefined}
                        loadingRejections={loadingRejections === run.run_id}
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Run row with expandable rejections ── */

function RunRow({
  run,
  expanded,
  onToggle,
  rejections: rejs,
  loadingRejections,
}: {
  run: DatasetRun;
  expanded: boolean;
  onToggle: () => void;
  rejections?: Rejection[];
  loadingRejections: boolean;
}) {
  const rate = run.acceptance_rate;
  const rateColor =
    rate >= 0.9
      ? "text-success"
      : rate >= 0.7
        ? "text-warning"
        : "text-error";

  return (
    <>
      <tr
        onClick={onToggle}
        className="border-t border-border hover:bg-surface-alt/50 cursor-pointer transition-colors"
      >
        <td className="px-4 py-3 font-mono text-xs text-brand-600">
          {run.run_id.slice(0, 12)}
        </td>
        <td className="px-4 py-3 text-text-secondary">
          {new Date(run.created_at).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </td>
        <td className="px-4 py-3 text-right text-success font-medium">
          {run.accepted}
        </td>
        <td className="px-4 py-3 text-right text-error font-medium">
          {run.rejected}
        </td>
        <td className="px-4 py-3 text-right text-text-primary font-medium">
          {run.total}
        </td>
        <td className={`px-4 py-3 text-right font-semibold ${rateColor}`}>
          {(rate * 100).toFixed(0)}%
        </td>
        <td className="px-4 py-3 text-right">
          <svg
            className={`w-4 h-4 text-text-muted transition-transform inline-block ${expanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m19.5 8.25-7.5 7.5-7.5-7.5"
            />
          </svg>
        </td>
      </tr>

      {expanded && (
        <tr>
          <td colSpan={7} className="px-4 py-4 bg-surface-alt/30">
            {loadingRejections && (
              <p className="text-xs text-text-muted">Loading rejections...</p>
            )}
            {rejs && rejs.length === 0 && (
              <p className="text-xs text-text-muted">No rejections.</p>
            )}
            {rejs && rejs.length > 0 && (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {rejs.map((r, i) => (
                  <div
                    key={`${r.scenario_id}-${i}`}
                    className="rounded-lg border border-border bg-surface px-4 py-3"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-mono text-text-muted mb-1">
                          {r.scenario_id}
                        </p>
                        <p className="text-sm text-text-primary">{r.reason}</p>
                        {r.details && (
                          <p className="text-xs text-text-secondary mt-1">
                            {r.details}
                          </p>
                        )}
                      </div>
                      {r.category && (
                        <span className="shrink-0 text-xs px-2 py-0.5 rounded-full bg-error/10 text-error font-medium">
                          {r.category}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
