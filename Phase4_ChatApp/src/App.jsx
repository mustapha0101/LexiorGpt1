import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, 
  Send, 
  Settings, 
  Activity, 
  FileText, 
  CheckCircle, 
  Sparkles, 
  AlertCircle, 
  RefreshCw, 
  Check, 
  Star,
  BookOpen
} from 'lucide-react';

// Custom Markdown-like Renderer to handle legal CoT formatting cleanly without dependencies
const formatMarkdown = (text) => {
  if (!text) return "";
  const lines = text.split("\n");
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const content = trimmed.substring(2);
      return <li key={idx} className="ml-5 list-disc pl-1 my-1 text-gray-300">{parseInline(content)}</li>;
    }
    const numMatch = trimmed.match(/^(\d+)\.\s(.*)/);
    if (numMatch) {
      return <li key={idx} className="ml-5 list-decimal pl-1 my-1 text-gray-300">{parseInline(numMatch[2])}</li>;
    }
    if (trimmed.startsWith("### ")) {
      return <h4 key={idx} className="text-lg font-semibold text-indigo-300 mt-4 mb-2">{parseInline(trimmed.substring(4))}</h4>;
    }
    if (trimmed.startsWith("## ")) {
      return <h3 key={idx} className="text-xl font-bold text-emerald-400 mt-5 mb-3">{parseInline(trimmed.substring(3))}</h3>;
    }
    if (!trimmed) {
      return <div key={idx} className="h-2"></div>;
    }
    return <p key={idx} className="my-2 leading-relaxed text-gray-200">{parseInline(line)}</p>;
  });
};

const parseInline = (text) => {
  const parts = text.split(/\*\*([^*]+)\*\*/g);
  return parts.map((part, i) => {
    if (i % 2 === 1) {
      return <strong key={i} className="text-emerald-400 font-semibold">{part}</strong>;
    }
    // Check for code blocks in line
    const codeParts = part.split(/`([^`]+)`/g);
    return codeParts.map((subPart, j) => {
      if (j % 2 === 1) {
        return <code key={j} className="bg-gray-800 text-pink-400 px-1 py-0.5 rounded font-mono text-sm">{subPart}</code>;
      }
      return subPart;
    });
  });
};

// Benchmark Legal Scenarios
const LEGAL_SCENARIOS = [
  {
    title: "Modification unilatérale du contrat (Droit du travail - Québec)",
    description: "Évalue si le modèle identifie les limites de l'employeur pour modifier substantiellement les conditions de travail.",
    prompt: "Un employeur au Québec peut-il modifier unilatéralement les heures de travail d'un employé sans son consentement ? Explique les étapes d'analyse juridique et la jurisprudence applicable.",
    criteria: [
      { text: "Vérifier la présence d'une clause contractuelle d'heures variables", met: false },
      { text: "Analyser la notion de modification substantielle des conditions", met: false },
      { text: "Identifier l'article 2085 du Code civil du Québec (Lien de subordination)", met: false },
      { text: "Évoquer la notion de congédiement déguisé en cas de refus", met: false },
      { text: "Mentionner les délais d'acceptation ou de contestation raisonnables", met: false }
    ]
  },
  {
    title: "Obligations PRP et Fuite SAAQclic (Droit administratif - Québec)",
    description: "Évalue la capacité du modèle à appliquer les cadres réglementaires en cas de compromission de données citoyennes.",
    prompt: "Dans le cadre du portail SAAQclic, quelles sont les obligations légales strictes de la SAAQ en cas d'accès non autorisé à des renseignements personnels de citoyens ? Quelles lois s'appliquent et quelles sont les démarches administratives requises ?",
    criteria: [
      { text: "Citer la Loi sur l'accès aux documents des organismes publics et sur la protection des renseignements personnels (Loi sur l'accès)", met: false },
      { text: "Mentionner l'évaluation obligatoire du préjudice sérieux", met: false },
      { text: "Identifier l'obligation d'avis écrit à la Commission d'accès à l'information (CAI)", met: false },
      { text: "Spécifier la notification obligatoire aux citoyens concernés", met: false },
      { text: "Mentionner la journalisation et les correctifs techniques obligatoires", met: false }
    ]
  },
  {
    title: "Tolérance Zéro Alcool & Permis Probatoire (Sécurité routière - Québec)",
    description: "Évalue la justesse normative sur les interdictions strictes et pénalités applicables aux jeunes conducteurs.",
    prompt: "Un conducteur de 20 ans sous le régime du permis probatoire au Québec peut-il légalement conduire avec un taux d'alcoolémie de 0,02 % ? Justifie juridiquement avec le Code de la sécurité routière.",
    criteria: [
      { text: "Citer l'application de la règle du double zéro (tolérance zéro alcool)", met: false },
      { text: "Identifier l'infraction à l'article 202.1 du Code de la sécurité routière (CSR)", met: false },
      { text: "Mentionner la suspension immédiate du permis de conduire pour 90 jours", met: false },
      { text: "Expliquer l'imposition de 4 points d'inaptitude et de l'amende", met: false },
      { text: "Mentionner que la règle s'applique à tous les titulaires de moins de 22 ans", met: false }
    ]
  }
];

export default function App() {
  // Connection Config State
  const [apiUrl, setApiUrl] = useState('https://6eys2nzfy3u10a-8000.proxy.runpod.net/v1');
  const [apiKey, setApiKey] = useState('none');
  const [modelId, setModelId] = useState('intelliwork/LexiorGpt1-merged');
  const [systemPrompt, setSystemPrompt] = useState(
    'You are LexiorGpt, a specialized legal AI assistant. Provide structured, step-by-step reasoning (Chain of Thought) in French.'
  );
  
  // Model Parameters State
  const [temperature, setTemperature] = useState(0.3);
  const [maxTokens, setMaxTokens] = useState(1200);

  // Layout Tab State
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'bench'

  // Chat State
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Bonjour ! Je suis **LexiorGpt**, votre modèle de raisonnement spécialisé en droit canadien. Entrez un prompt ou rendez-vous sur l'onglet **Benchmark** pour évaluer mes capacités." }
  ]);
  const [currentInput, setCurrentInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('idle'); // 'idle', 'testing', 'ready', 'error'

  // Telemetry & Metrics State
  const [latency, setLatency] = useState(0); // in ms
  const [tokensPerSec, setTokensPerSec] = useState(0);
  const [inputTokens, setInputTokens] = useState(0);
  const [outputTokens, setOutputTokens] = useState(0);
  const [speedHistory, setSpeedHistory] = useState([35, 45, 52, 48, 55, 53, 52]);

  // Benchmarks State
  const [scenarios, setScenarios] = useState(LEGAL_SCENARIOS);
  const [selectedScenarioIdx, setSelectedScenarioIdx] = useState(0);
  const [benchGrades, setBenchGrades] = useState({}); // { [idx]: { accuracy: 5, reasoning: 4, structure: 5 } }

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle Test Connection
  const testConnection = async () => {
    setConnectionStatus('testing');
    try {
      const response = await fetch(`${apiUrl}/models`, {
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });
      if (response.ok) {
        setConnectionStatus('ready');
      } else {
        setConnectionStatus('error');
      }
    } catch {
      setConnectionStatus('error');
    }
  };

  // Run Inference / Stream response
  const handleSend = async (customPrompt = null) => {
    const promptText = customPrompt || currentInput;
    if (!promptText.trim() || isStreaming) return;

    if (!customPrompt) {
      setCurrentInput('');
    }

    const newMessages = [
      ...messages,
      { role: 'user', content: promptText }
    ];
    setMessages([...newMessages, { role: 'assistant', content: '' }]);
    setIsStreaming(true);

    const startTime = Date.now();
    let firstTokenTime = 0;
    let tokenCount = 0;

    try {
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: modelId,
          messages: [
            { role: 'system', content: systemPrompt },
            ...newMessages
          ],
          temperature: temperature,
          max_tokens: maxTokens,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error("L'API a retourné un code d'erreur.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let assistantContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          const cleanedLine = line.trim();
          if (cleanedLine.startsWith('data: ')) {
            if (cleanedLine.includes('[DONE]')) break;
            try {
              const data = JSON.parse(cleanedLine.slice(6));
              const textChunk = data.choices[0]?.delta?.content || '';
              
              if (textChunk) {
                if (tokenCount === 0) {
                  firstTokenTime = Date.now();
                  setLatency(firstTokenTime - startTime);
                }
                tokenCount += 1;
                assistantContent += textChunk;

                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: 'assistant', content: assistantContent };
                  return updated;
                });

                // Calculate speed
                const elapsedSec = (Date.now() - firstTokenTime) / 1000;
                if (elapsedSec > 0.1) {
                  const speed = Math.round(tokenCount / elapsedSec);
                  setTokensPerSec(speed);
                  setSpeedHistory(prev => [...prev.slice(-10), speed]);
                }
              }
            } catch (e) {
              // Ignore JSON parse errors on incomplete lines
            }
          }
        }
      }

      // Finalize counts
      setInputTokens(Math.round(promptText.length / 3.8));
      setOutputTokens(tokenCount);
      setConnectionStatus('ready');

    } catch (err) {
      console.error(err);
      setConnectionStatus('error');
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { 
          role: 'assistant', 
          content: `❌ **Erreur d'inférence** : Impossible de contacter le modèle vLLM sur [${apiUrl}]. Vérifiez que votre Pod est actif et que l'URL est correcte.` 
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  // Toggle criteria checkbox
  const toggleCriterion = (scenarioIdx, critIdx) => {
    const updated = [...scenarios];
    updated[scenarioIdx].criteria[critIdx].met = !updated[scenarioIdx].criteria[critIdx].met;
    setScenarios(updated);
  };

  // Save scenario grades
  const submitGrade = (scenarioIdx, grades) => {
    setBenchGrades(prev => ({
      ...prev,
      [scenarioIdx]: grades
    }));
  };

  // Calculate average session rating
  const getAverageScore = () => {
    const keys = Object.keys(benchGrades);
    if (keys.length === 0) return 'Non noté';
    let total = 0;
    keys.forEach(k => {
      const g = benchGrades[k];
      total += (g.accuracy + g.reasoning + g.structure) / 3;
    });
    return (total / keys.length).toFixed(1) + ' / 5.0';
  };

  return (
    <div className="app-container">
      {/* 1. LEFT COLUMN: Configuration Sidebar (350px) */}
      <aside className="w-[350px] glass-panel border-r border-gray-800 flex flex-col h-full overflow-y-auto">
        <div className="p-5 border-b border-gray-800 flex items-center gap-3">
          <Sparkles className="text-emerald-400 w-6 h-6 animate-pulse" />
          <h1 className="text-lg font-bold tracking-tight text-white">LexiorGPT Console</h1>
        </div>

        {/* API connection parameters */}
        <div className="p-5 border-b border-gray-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
              <Settings className="w-4 h-4" /> Paramètres d'API
            </h3>
            <span className={`w-2.5 h-2.5 rounded-full ${
              connectionStatus === 'ready' ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' :
              connectionStatus === 'testing' ? 'bg-indigo-500 animate-pulse' :
              connectionStatus === 'error' ? 'bg-red-500 shadow-[0_0_10px_#ef4444]' : 'bg-gray-500'
            }`}></span>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-gray-400">Endpoint URL (RunPod vLLM)</label>
            <input 
              type="text" 
              className="glass-input text-xs" 
              value={apiUrl} 
              onChange={(e) => setApiUrl(e.target.value)} 
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-gray-400">Hugging Face Token / API Key</label>
            <input 
              type="password" 
              className="glass-input text-xs" 
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)} 
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-gray-400">Model ID</label>
            <input 
              type="text" 
              className="glass-input text-xs" 
              value={modelId} 
              onChange={(e) => setModelId(e.target.value)} 
            />
          </div>

          <button onClick={testConnection} className="glass-button w-full text-xs justify-center active">
            <RefreshCw className="w-3.5 h-3.5" /> Tester la Connexion
          </button>
        </div>

        {/* Hyperparameters Config */}
        <div className="p-5 border-b border-gray-800 flex flex-col gap-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Configuration Inférence</h3>
          
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Température</span>
              <span className="text-emerald-400 font-semibold">{temperature}</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              className="accent-emerald-500" 
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Max Tokens</span>
              <span className="text-indigo-400 font-semibold">{maxTokens}</span>
            </div>
            <input 
              type="range" 
              min="100" 
              max="4096" 
              step="50"
              className="accent-indigo-500" 
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* System Prompt Config */}
        <div className="p-5 flex-1 flex flex-col gap-2">
          <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">System Prompt (Instructions)</label>
          <textarea 
            className="glass-input flex-1 resize-none text-xs leading-relaxed" 
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>
      </aside>

      {/* 2. RIGHT COLUMN: Interactive Workspaces */}
      <main className="flex-1 flex flex-col h-full">
        {/* Top Header Navigation Tabs */}
        <header className="h-[60px] glass-panel border-b border-gray-800 px-6 flex items-center justify-between">
          <div className="flex gap-4">
            <button 
              onClick={() => setActiveTab('chat')} 
              className={`glass-button text-sm ${activeTab === 'chat' ? 'active' : ''}`}
            >
              <Activity className="w-4 h-4" /> Playground & Performance
            </button>
            <button 
              onClick={() => setActiveTab('bench')} 
              className={`glass-button text-sm ${activeTab === 'bench' ? 'active' : ''}`}
            >
              <FileText className="w-4 h-4" /> Benchmark d'Évaluation CoT
            </button>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-400">Note Générale Session :</span>
            <span className="bg-emerald-500/10 text-emerald-400 font-bold px-3 py-1 rounded-full border border-emerald-500/20 text-xs">
              {getAverageScore()}
            </span>
          </div>
        </header>

        {/* WORKSPACE 1: CHAT & PERFORMANCE WORKSPACE */}
        {activeTab === 'chat' && (
          <div className="flex-1 flex overflow-hidden">
            {/* Chat Feed Column (60%) */}
            <div className="flex-1 flex flex-col border-r border-gray-800 h-full bg-black/10">
              <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
                {messages.map((msg, i) => (
                  <div 
                    key={i} 
                    className={`flex flex-col max-w-[80%] animate-slide-in ${
                      msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'
                    }`}
                  >
                    <span className="text-[10px] text-gray-400 mb-1 px-1">
                      {msg.role === 'user' ? 'Utilisateur' : 'LexiorGPT (vLLM)'}
                    </span>
                    <div className={`p-4 rounded-2xl ${
                      msg.role === 'user' 
                        ? 'bg-gradient-to-r from-emerald-600 to-teal-700 text-white rounded-br-none shadow-lg' 
                        : 'glass-panel border border-gray-800 text-gray-100 rounded-bl-none shadow-xl'
                    }`}>
                      {formatMarkdown(msg.content)}
                    </div>
                  </div>
                ))}
                
                {isStreaming && (
                  <div className="flex flex-col max-w-[80%] self-start items-start animate-slide-in">
                    <span className="text-[10px] text-gray-400 mb-1 px-1">LexiorGPT (vLLM)</span>
                    <div className="p-4 rounded-2xl glass-panel border border-gray-800 text-gray-100 rounded-bl-none flex items-center gap-1.5">
                      <div className="dot"></div>
                      <div className="dot"></div>
                      <div className="dot"></div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Bar */}
              <div className="p-4 border-t border-gray-800 flex gap-3 bg-black/25">
                <input 
                  type="text" 
                  className="glass-input flex-1 text-sm py-3" 
                  placeholder="Posez votre question juridique (Ex: Règle d'indemnité de préavis...)" 
                  value={currentInput}
                  onChange={(e) => setCurrentInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  disabled={isStreaming}
                />
                <button 
                  onClick={() => handleSend()} 
                  className="glass-button primary py-3 px-5" 
                  disabled={isStreaming}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Performance Panel Column (40%) */}
            <div className="w-[380px] p-6 flex flex-col gap-6 overflow-y-auto h-full">
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" /> Télémétrie en Temps Réel
              </h3>

              {/* Grid Widgets */}
              <div className="grid grid-cols-2 gap-4">
                <div className="glass-panel p-4 rounded-xl flex flex-col">
                  <span className="text-xs text-gray-400">Latence Premier Token</span>
                  <span className="text-2xl font-bold text-white mt-1">
                    {latency > 0 ? `${latency} ms` : '-'}
                  </span>
                  <span className="text-[10px] text-indigo-400 mt-0.5">Calculé sur l'API vLLM</span>
                </div>

                <div className="glass-panel p-4 rounded-xl flex flex-col">
                  <span className="text-xs text-gray-400">Vitesse Génération</span>
                  <span className="text-2xl font-bold text-white mt-1">
                    {tokensPerSec > 0 ? `${tokensPerSec} t/s` : '-'}
                  </span>
                  <span className="text-[10px] text-emerald-400 mt-0.5">Vitesse de streaming</span>
                </div>
              </div>

              {/* Token Usage Widget */}
              <div className="glass-panel p-4 rounded-xl flex flex-col gap-2">
                <span className="text-xs text-gray-400">Consommation des Tokens</span>
                <div className="flex justify-between items-center mt-1 border-b border-gray-800/50 pb-2">
                  <span className="text-xs text-gray-400">Tokens d'Entrée (Prompt) :</span>
                  <span className="text-sm font-semibold text-gray-200">{inputTokens}</span>
                </div>
                <div className="flex justify-between items-center border-b border-gray-800/50 pb-2">
                  <span className="text-xs text-gray-400">Tokens de Sortie (Générés) :</span>
                  <span className="text-sm font-semibold text-gray-200">{outputTokens}</span>
                </div>
                <div className="flex justify-between items-center pt-1">
                  <span className="text-xs font-bold text-gray-300">Total :</span>
                  <span className="text-sm font-bold text-indigo-400">{inputTokens + outputTokens}</span>
                </div>
              </div>

              {/* Performance SVG Line Chart */}
              <div className="glass-panel p-4 rounded-xl flex flex-col gap-2">
                <span className="text-xs text-gray-400 mb-2">Historique de Vitesse (tokens/sec)</span>
                <div className="w-full h-[120px] bg-black/20 rounded-lg flex items-end p-2 relative overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 100 50" preserveAspectRatio="none">
                    <polyline
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2"
                      points={speedHistory.map((val, idx) => {
                        const x = (idx / (speedHistory.length - 1)) * 100;
                        const y = 50 - (val / 100) * 50; // Normalize relative to 100 max speed
                        return `${x},${y}`;
                      }).join(' ')}
                    />
                  </svg>
                  <div className="absolute right-2 top-2 bg-indigo-500/10 text-[9px] text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20">
                    Max: 100 t/s
                  </div>
                </div>
              </div>

              {/* Hardware diagnostics */}
              <div className="glass-panel p-4 rounded-xl flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-indigo-400" />
                <div className="flex flex-col">
                  <span className="text-xs font-semibold text-white">Modèle Qwen-32B non quantifié</span>
                  <span className="text-[10px] text-gray-400 leading-relaxed">
                    Exécution en FP16 natif. Nécessite 80 Go de VRAM. KV Cache optimisé par vLLM.
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 2: BENCHMARK EVALUATION WORKSPACE */}
        {activeTab === 'bench' && (
          <div className="flex-1 flex overflow-hidden">
            {/* Scenarios Panel (30%) */}
            <div className="w-[320px] border-r border-gray-800 h-full flex flex-col bg-black/10">
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-emerald-400" /> Scénarios de Test
                </h3>
              </div>
              <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
                {scenarios.map((sc, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedScenarioIdx(idx)}
                    className={`text-left p-4 rounded-xl transition-all border ${
                      selectedScenarioIdx === idx 
                        ? 'glass-panel border-emerald-500/30 bg-emerald-500/5' 
                        : 'border-transparent hover:bg-white/5'
                    }`}
                  >
                    <h4 className={`text-xs font-bold ${selectedScenarioIdx === idx ? 'text-emerald-400' : 'text-gray-200'}`}>
                      {sc.title}
                    </h4>
                    <p className="text-[10px] text-gray-400 mt-1 line-clamp-2 leading-relaxed">
                      {sc.description}
                    </p>
                  </button>
                ))}
              </div>
            </div>

            {/* Run & Verification Workspace (70%) */}
            <div className="flex-1 flex flex-col h-full overflow-y-auto p-6 gap-6">
              <div className="glass-panel p-6 rounded-2xl flex flex-col gap-4">
                <div className="flex justify-between items-start gap-4">
                  <div>
                    <h2 className="text-xl font-bold text-white">{scenarios[selectedScenarioIdx].title}</h2>
                    <p className="text-xs text-gray-400 mt-1">{scenarios[selectedScenarioIdx].description}</p>
                  </div>
                  <button 
                    onClick={() => handleSend(scenarios[selectedScenarioIdx].prompt)}
                    className="glass-button primary" 
                    disabled={isStreaming}
                  >
                    <Play className="w-4 h-4" /> Lancer l'Évaluation
                  </button>
                </div>
                
                <div className="bg-black/25 p-4 rounded-lg border border-gray-800/80">
                  <span className="text-[10px] text-emerald-400 uppercase font-bold tracking-wider">Prompt envoyé :</span>
                  <p className="text-xs text-gray-300 mt-1 leading-relaxed">{scenarios[selectedScenarioIdx].prompt}</p>
                </div>
              </div>

              {/* Grid for criteria checklist & grading */}
              <div className="grid grid-cols-5 gap-6">
                
                {/* Criteria checklist (3 columns) */}
                <div className="col-span-3 glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <CheckCircle className="w-4.5 h-4.5 text-emerald-400" /> Critères de CoT Attendus
                  </h3>
                  <div className="flex flex-col gap-3">
                    {scenarios[selectedScenarioIdx].criteria.map((crit, cIdx) => (
                      <label 
                        key={cIdx} 
                        className={`flex items-start gap-3 p-3 rounded-lg border transition-all cursor-pointer ${
                          crit.met 
                            ? 'bg-emerald-500/5 border-emerald-500/20 text-gray-200' 
                            : 'border-gray-800 hover:border-gray-700 text-gray-400'
                        }`}
                      >
                        <input 
                          type="checkbox" 
                          checked={crit.met} 
                          onChange={() => toggleCriterion(selectedScenarioIdx, cIdx)}
                          className="mt-0.5 accent-emerald-500"
                        />
                        <span className="text-xs leading-relaxed">{crit.text}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Performance score grading (2 columns) */}
                <div className="col-span-2 glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Star className="w-4.5 h-4.5 text-yellow-400" /> Évaluation Qualitative
                  </h3>
                  
                  <GradeSlider 
                    label="Exactitude Juridique" 
                    value={benchGrades[selectedScenarioIdx]?.accuracy || 5} 
                    onChange={(val) => submitGrade(selectedScenarioIdx, {
                      ...(benchGrades[selectedScenarioIdx] || { accuracy: 5, reasoning: 5, structure: 5 }),
                      accuracy: val
                    })}
                  />

                  <GradeSlider 
                    label="Clarté du Raisonnement (CoT)" 
                    value={benchGrades[selectedScenarioIdx]?.reasoning || 5} 
                    onChange={(val) => submitGrade(selectedScenarioIdx, {
                      ...(benchGrades[selectedScenarioIdx] || { accuracy: 5, reasoning: 5, structure: 5 }),
                      reasoning: val
                    })}
                  />

                  <GradeSlider 
                    label="Structure & Format" 
                    value={benchGrades[selectedScenarioIdx]?.structure || 5} 
                    onChange={(val) => submitGrade(selectedScenarioIdx, {
                      ...(benchGrades[selectedScenarioIdx] || { accuracy: 5, reasoning: 5, structure: 5 }),
                      structure: val
                    })}
                  />

                  <div className="bg-yellow-500/5 border border-yellow-500/10 p-3 rounded-lg flex items-start gap-2.5 mt-2">
                    <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5" />
                    <span className="text-[10px] text-yellow-500/80 leading-relaxed">
                      L'évaluation qualitative permet de mesurer la pertinence des explications étape par étape et l'alignement sur les politiques de l'entreprise.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Simple internal helper component for star rating / slider
function GradeSlider({ label, value, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="text-yellow-400 font-bold">{value} / 5</span>
      </div>
      <div className="flex gap-1.5 mt-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => onChange(star)}
            className={`p-1 rounded transition-all ${
              star <= value ? 'text-yellow-400' : 'text-gray-600'
            }`}
          >
            <Star className="w-5 h-5 fill-current" />
          </button>
        ))}
      </div>
    </div>
  );
}
