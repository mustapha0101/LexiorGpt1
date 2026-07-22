const STEPS = [
  { node: "plan", label: "Planning" },
  { node: "execute_tool", label: "Tool Execution" },
  { node: "generate_answer", label: "Generating Answer" },
  { node: "run_critics", label: "Quality Review" },
  { node: "repair", label: "Repair" },
  { node: "validate_final", label: "Validation" },
  { node: "export", label: "Export" },
] as const;

interface Props {
  currentNode: string | null;
  visitedNodes: string[];
}

export function AgentProgress({ currentNode, visitedNodes }: Props) {
  if (!currentNode && visitedNodes.length === 0) return null;

  return (
    <div className="flex items-center gap-1 px-3 py-2 mb-2 rounded-xl bg-surface-alt border border-border overflow-x-auto">
      {STEPS.map((step, i) => {
        const isCurrent = currentNode === step.node;
        const isVisited = visitedNodes.includes(step.node);
        const isRejected =
          currentNode === "reject" && !isVisited && !isCurrent;

        let dotClass =
          "w-2.5 h-2.5 rounded-full shrink-0 transition-all duration-300";
        let labelClass = "text-[11px] whitespace-nowrap transition-colors";

        if (isCurrent) {
          dotClass += " bg-brand-500 ring-2 ring-brand-300 animate-pulse";
          labelClass += " text-brand-600 font-semibold";
        } else if (isVisited) {
          dotClass += " bg-brand-400";
          labelClass += " text-text-secondary";
        } else if (isRejected) {
          dotClass += " bg-surface-raised";
          labelClass += " text-text-muted/40";
        } else {
          dotClass += " bg-surface-raised";
          labelClass += " text-text-muted";
        }

        return (
          <div key={step.node} className="flex items-center gap-1">
            {i > 0 && (
              <div
                className={`w-4 h-px mx-0.5 ${
                  isVisited || isCurrent
                    ? "bg-brand-400"
                    : "bg-border"
                }`}
              />
            )}
            <div className="flex items-center gap-1">
              <span className={dotClass} />
              <span className={labelClass}>{step.label}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
