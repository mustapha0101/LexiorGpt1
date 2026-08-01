import { useCallback, useMemo, useState } from "react";
import type {
  HumanEvaluationMode,
  HumanEvaluationRun,
  HumanScenario,
} from "../types";

export interface EvaluationChatContext {
  mode: "human_40";
  runId: string;
  scenarioId: number;
  threadId: string;
}

export interface HumanEvaluationReturn {
  mode: HumanEvaluationMode;
  setMode: (mode: HumanEvaluationMode) => void;
  run: HumanEvaluationRun | null;
  currentScenario: HumanScenario | null;
  saving: boolean;
  saveError: string | null;
  startNewEvaluation: () => Promise<HumanEvaluationRun>;
  resumeEvaluation: (runId: string) => Promise<HumanEvaluationRun>;
  startScenario: (scenarioId: number) => Promise<EvaluationChatContext>;
  saveReview: (scenarioId: number, review: Record<string, unknown>) => Promise<void>;
  completeScenario: (scenarioId: number) => Promise<void>;
  interruptScenario: (scenarioId: number) => Promise<void>;
  completeRun: () => Promise<void>;
  openFile: () => void;
}

const MODE_KEY = "lexior-evaluation-mode";

function readMode(): HumanEvaluationMode {
  return localStorage.getItem(MODE_KEY) === "human_40" ? "human_40" : "normal";
}

async function jsonRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(String(body.detail ?? `HTTP ${response.status}`));
  }
  return body as T;
}

export function useHumanEvaluation(): HumanEvaluationReturn {
  const [mode, setModeState] = useState<HumanEvaluationMode>(readMode);
  const [run, setRun] = useState<HumanEvaluationRun | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const setMode = useCallback((next: HumanEvaluationMode) => {
    setModeState(next);
    localStorage.setItem(MODE_KEY, next);
  }, []);

  const withSave = useCallback(async <T,>(operation: () => Promise<T>) => {
    setSaving(true);
    setSaveError(null);
    try {
      return await operation();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Erreur de sauvegarde";
      setSaveError(message);
      throw error;
    } finally {
      setSaving(false);
    }
  }, []);

  const startNewEvaluation = useCallback(
    () => withSave(async () => {
      const created = await jsonRequest<HumanEvaluationRun>(
        "/api/evaluations/human-40/runs", { method: "POST", body: "{}" });
      const started = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(created.run.run_id)}/scenarios/1/start`,
        { method: "POST", body: "{}" });
      setRun(started);
      setMode("human_40");
      return started;
    }),
    [setMode, withSave],
  );

  const resumeEvaluation = useCallback(
    (runId: string) => withSave(async () => {
      const loaded = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(runId)}`);
      const current = loaded.run.current_scenario_id
        ? loaded.scenarios.find((item) => item.scenario_id === loaded.run.current_scenario_id)
        : null;
      const resumed = current?.status === "interrupted"
        ? await jsonRequest<HumanEvaluationRun>(
          `/api/evaluations/human-40/runs/${encodeURIComponent(runId)}/scenarios/${current.scenario_id}/start`,
          { method: "POST", body: "{}" })
        : loaded;
      setRun(resumed);
      setMode("human_40");
      return resumed;
    }),
    [setMode, withSave],
  );

  const startScenario = useCallback(
    (scenarioId: number) => withSave(async () => {
      if (!run) throw new Error("Commencez ou reprenez une evaluation");
      const updated = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(run.run.run_id)}/scenarios/${scenarioId}/start`,
        { method: "POST", body: JSON.stringify({}) });
      setRun(updated);
      const scenario = updated.scenarios.find((item) => item.scenario_id === scenarioId);
      if (!scenario?.thread_id) throw new Error("Thread d'evaluation absent");
      return {
        mode: "human_40" as const,
        runId: updated.run.run_id,
        scenarioId,
        threadId: scenario.thread_id,
      };
    }),
    [run, withSave],
  );

  const saveReview = useCallback(
    (scenarioId: number, review: Record<string, unknown>) => withSave(async () => {
      const updated = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(run!.run.run_id)}/scenarios/${scenarioId}`,
        { method: "PATCH", body: JSON.stringify(review) });
      setRun(updated);
    }),
    [run, withSave],
  );

  const completeScenario = useCallback(
    (scenarioId: number) => withSave(async () => {
      const updated = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(run!.run.run_id)}/scenarios/${scenarioId}/complete`,
        { method: "POST", body: "{}" });
      setRun(updated);
    }),
    [run, withSave],
  );

  const interruptScenario = useCallback(
    (scenarioId: number) => withSave(async () => {
      const updated = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(run!.run.run_id)}`,
        { method: "PATCH", body: JSON.stringify({ action: "interrupt_scenario", scenario_id: scenarioId }) });
      setRun(updated);
    }),
    [run, withSave],
  );

  const completeRun = useCallback(
    () => withSave(async () => {
      const updated = await jsonRequest<HumanEvaluationRun>(
        `/api/evaluations/human-40/runs/${encodeURIComponent(run!.run.run_id)}/complete`,
        { method: "POST", body: "{}" });
      setRun(updated);
    }),
    [run, withSave],
  );

  const openFile = useCallback(() => {
    if (run) window.open(
      `/api/evaluations/human-40/runs/${encodeURIComponent(run.run.run_id)}/file`,
      "_blank", "noopener,noreferrer");
  }, [run]);

  const currentScenario = useMemo(
    () => run?.scenarios.find((item) => item.scenario_id === run.run.current_scenario_id) ?? null,
    [run],
  );

  return {
    mode, setMode, run, currentScenario, saving, saveError,
    startNewEvaluation, resumeEvaluation, startScenario, saveReview,
    completeScenario, interruptScenario, completeRun, openFile,
  };
}
