import { useCallback, useState } from "react";
import type { AppView } from "./types";
import { useChat } from "./hooks/useChat";
import { useHumanEvaluation } from "./hooks/useHumanEvaluation";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";
import { Dashboard } from "./components/Dashboard";

export default function App() {
  const [view, setView] = useState<AppView>("chat");
  const evaluation = useHumanEvaluation();
  const chat = useChat();

  const activateScenario = useCallback(async (scenarioId: number) => {
    if (evaluation.currentScenario
        && evaluation.currentScenario.scenario_id !== scenarioId
        && evaluation.currentScenario.status === "in_progress") {
      await evaluation.interruptScenario(evaluation.currentScenario.scenario_id);
    }
    const context = await evaluation.startScenario(scenarioId);
    chat.beginThread(context.threadId);
    chat.setEvaluationContext(context);
  }, [chat, evaluation]);

  const loadEvaluationIntoChat = useCallback((loaded: typeof evaluation.run) => {
    if (!loaded) return;
    const scenarioId = loaded?.run.current_scenario_id;
    const scenario = scenarioId
      ? loaded.scenarios.find((item) => item.scenario_id === scenarioId)
      : null;
    if (scenario?.thread_id) {
      const context = {
        mode: "human_40" as const,
        runId: loaded.run.run_id,
        scenarioId: scenario.scenario_id,
        threadId: scenario.thread_id,
      };
      chat.beginThread(scenario.thread_id);
      chat.setEvaluationContext(context);
    }
  }, [chat]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar
        currentView={view}
        onNavigate={setView}
        agentLog={chat.agentLog}
        rawEvents={chat.rawEvents}
        streaming={chat.streaming}
      />

      <main className="flex-1 min-w-0">
        {view === "chat" ? (
          <Chat
            chat={chat}
            evaluation={evaluation}
            onStartScenario={activateScenario}
            onEvaluationLoaded={loadEvaluationIntoChat}
          />
        ) : <Dashboard />}
      </main>
    </div>
  );
}
