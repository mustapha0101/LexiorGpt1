import { useState } from "react";
import type { HumanEvaluationReturn } from "../hooks/useHumanEvaluation";
import type { HumanScenario } from "../types";

interface Props {
  evaluation: HumanEvaluationReturn;
  canReview: boolean;
  onStartScenario: (scenarioId: number) => Promise<void>;
  onEvaluationLoaded: (run: HumanEvaluationReturn["run"]) => void;
}

const RESPONSE_QUALITY: Array<[string, string]> = [
  ["answers_question", "Répond à la question"],
  ["useful", "Utile"],
  ["too_general", "Trop générale"],
  ["too_cautious", "Trop prudente"],
  ["too_affirmative", "Trop affirmative"],
  ["unsupported_rule", "Règle non fondée"],
  ["wrong_application", "Application incorrecte"],
  ["insufficient_citations", "Citations insuffisantes"],
  ["useful_clarification", "Clarification utile"],
  ["repeated_clarification", "Clarification répétée"],
  ["technical_error", "Erreur technique"],
  ["slow", "Lente"],
];

export function HumanEvaluationPanel({ evaluation, canReview, onStartScenario, onEvaluationLoaded }: Props) {
  const [resumeId, setResumeId] = useState("");
  const [expected, setExpected] = useState("");
  const [rating, setRating] = useState("");
  const [route, setRoute] = useState("");
  const [quality, setQuality] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [observed, setObserved] = useState("");
  const scenario = evaluation.currentScenario;

  async function beginNew() {
    try {
      const started = await evaluation.startNewEvaluation();
      onEvaluationLoaded(started);
    } catch { /* The hook exposes the save error. */ }
  }

  async function resume() {
    if (!resumeId.trim()) return;
    try {
      const loaded = await evaluation.resumeEvaluation(resumeId.trim());
      onEvaluationLoaded(loaded);
    } catch { /* The hook exposes the save error. */ }
  }

  async function saveAndComplete() {
    if (!scenario || !expected.trim() || !rating) return;
    try {
      await evaluation.saveReview(scenario.scenario_id, {
        expected_answer_note: expected.trim(),
        overall_rating: rating,
        route_quality: route || undefined,
        response_quality: quality,
        notes,
        observed_category: observed || undefined,
      });
      await evaluation.completeScenario(scenario.scenario_id);
      setExpected(""); setRating(""); setRoute(""); setQuality([]); setNotes(""); setObserved("");
    } catch { /* The hook exposes the save error. */ }
  }

  function toggleQuality(value: string) {
    setQuality((current) => current.includes(value)
      ? current.filter((item) => item !== value)
      : [...current, value]);
  }

  return (
    <section className="eval-panel border-t border-border bg-surface-alt px-6 py-3">
      <div className="max-w-3xl mx-auto space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <span className="text-xs font-semibold text-brand-700">Évaluation des 40 situations</span>
            {evaluation.run && scenario && (
              <p className="text-xs text-text-muted">
                Exécution : {evaluation.run.run.run_id} · Situation : {scenario.scenario_id} / 40 · Catégorie : {scenario.category} — {scenario.category_label} · Statut : {scenario.status}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {!evaluation.run && <button className="eval-button" onClick={beginNew}>Commencer une nouvelle évaluation</button>}
            {evaluation.run && <button className="eval-button" onClick={evaluation.openFile}>Exporter / ouvrir le fichier</button>}
            {evaluation.run && <button className="eval-button" onClick={() => evaluation.completeRun()}>Terminer l’évaluation</button>}
          </div>
        </div>

        {!evaluation.run && (
          <div className="flex flex-wrap items-center gap-2">
            <input value={resumeId} onChange={(event) => setResumeId(event.target.value)} placeholder="run_id à reprendre" className="eval-input" />
            <button className="eval-button" onClick={resume}>Reprendre une évaluation</button>
          </div>
        )}

        {evaluation.run && !scenario && evaluation.run.run.status === "in_progress" && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-text-muted">Choisissez la prochaine situation à évaluer.</span>
            <select className="eval-input" defaultValue="" onChange={(event) => event.target.value && onStartScenario(Number(event.target.value))}>
              <option value="">Situation suivante…</option>
              {evaluation.run.scenarios.filter((item) => item.status !== "completed").map((item) => (
                <option key={item.scenario_id} value={item.scenario_id}>Situation {item.scenario_id}</option>
              ))}
            </select>
          </div>
        )}

        {evaluation.run && scenario && (
          <>
            <div className="rounded-lg border border-border bg-surface px-3 py-2">
              <p className="text-xs text-text-muted">Rappel de la référence — elle n’est pas envoyée à LexiorGPT.</p>
              <p className="text-sm text-text-secondary">{scenario.scenario_description}</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <select value={scenario.scenario_id} onChange={(event) => onStartScenario(Number(event.target.value))} className="eval-input">
                {evaluation.run.scenarios.map((item) => <option key={item.scenario_id} value={item.scenario_id}>Situation {item.scenario_id} — {item.status}</option>)}
              </select>
              <button className="eval-button" onClick={() => onStartScenario(Math.max(1, scenario.scenario_id - 1))}>← Précédente</button>
              <button className="eval-button" onClick={() => onStartScenario(Math.min(40, scenario.scenario_id + 1))}>Suivante →</button>
              <button className="eval-button" onClick={() => evaluation.interruptScenario(scenario.scenario_id)}>Marquer interrompue</button>
            </div>
          </>
        )}

        {canReview && scenario && (
          <div className="rounded-xl border border-brand-200 bg-surface p-4 space-y-3">
            <p className="text-sm font-semibold text-text-primary">Évaluation humaine de cette réponse</p>
            <label className="eval-label">Qu’est-ce que tu attendais comme réponse? <textarea required value={expected} onChange={(event) => setExpected(event.target.value)} className="eval-textarea" /></label>
            <div className="grid sm:grid-cols-2 gap-3">
              <label className="eval-label">Évaluation globale<select value={rating} onChange={(event) => setRating(event.target.value)} className="eval-input"><option value="">Choisir…</option><option value="successful">Réussi</option><option value="partially_successful">Partiellement réussi</option><option value="failed">Échoué</option><option value="unable_to_judge">Impossible à juger</option></select></label>
              <label className="eval-label">Qualité de la route<select value={route} onChange={(event) => setRoute(event.target.value)} className="eval-input"><option value="">Choisir…</option><option>Bonne source et bonne route</option><option>Bonne réponse malgré une route différente</option><option>Source manquante</option><option>Mauvaise source</option><option>Recherche inutile</option><option>Trop de recherches</option><option>Impossible à juger</option></select></label>
            </div>
            <div><p className="eval-label">Qualité de la réponse</p><div className="grid sm:grid-cols-3 gap-1">{RESPONSE_QUALITY.map(([value, label]) => <label key={value} className="text-xs text-text-secondary"><input type="checkbox" checked={quality.includes(value)} onChange={() => toggleQuality(value)} /> {label}</label>)}</div></div>
            <label className="eval-label">Catégorie observée<select value={observed} onChange={(event) => setObserved(event.target.value)} className="eval-input"><option value="">Choisir…</option><option>Catégorie initiale confirmée</option><option>Article seul aurait suffi</option><option>Règlement seul nécessaire</option><option>Article et règlement nécessaires</option><option>Jurisprudence nécessaire</option><option>Autre route nécessaire</option><option>Impossible à déterminer</option></select></label>
            <label className="eval-label">Notes supplémentaires<textarea value={notes} onChange={(event) => setNotes(event.target.value)} className="eval-textarea" /></label>
            <button disabled={!expected.trim() || !rating || evaluation.saving} onClick={saveAndComplete} className="eval-primary">Enregistrer et terminer cette situation</button>
          </div>
        )}
        <div className="flex items-center gap-2 text-xs text-text-muted"><span>{evaluation.saving ? "Sauvegarde…" : evaluation.saveError ? "Erreur de sauvegarde — nouvelle tentative" : evaluation.run ? "Sauvegardé" : ""}</span></div>
      </div>
    </section>
  );
}

export function evaluationScenarioKey(scenario: HumanScenario | null): string {
  return scenario ? `${scenario.scenario_id}:${scenario.thread_id ?? ""}` : "none";
}
