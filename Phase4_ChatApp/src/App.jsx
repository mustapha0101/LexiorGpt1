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
  BookOpen,
  Lock,
  Unlock
} from 'lucide-react';

// Custom Markdown-like Renderer to handle legal CoT formatting cleanly without dependencies
const formatMarkdown = (text) => {
  if (!text) return "";
  const lines = text.split("\n");
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const content = trimmed.substring(2);
      return <li key={idx} style={{ marginLeft: '20px', listStyleType: 'disc', paddingLeft: '4px', margin: '4px 0', color: '#d1d5db' }}>{parseInline(content)}</li>;
    }
    const numMatch = trimmed.match(/^(\d+)\.\s(.*)/);
    if (numMatch) {
      return <li key={idx} style={{ marginLeft: '20px', listStyleType: 'decimal', paddingLeft: '4px', margin: '4px 0', color: '#d1d5db' }}>{parseInline(numMatch[2])}</li>;
    }
    if (trimmed.startsWith("### ")) {
      return <h4 key={idx} style={{ fontSize: '15px', fontWeight: '600', color: '#818cf8', marginTop: '16px', marginBottom: '8px' }}>{parseInline(trimmed.substring(4))}</h4>;
    }
    if (trimmed.startsWith("## ")) {
      return <h3 key={idx} style={{ fontSize: '18px', fontWeight: '700', color: '#34d399', marginTop: '20px', marginBottom: '12px' }}>{parseInline(trimmed.substring(3))}</h3>;
    }
    if (!trimmed) {
      return <div key={idx} style={{ height: '8px' }}></div>;
    }
    return <p key={idx} style={{ margin: '8px 0', lineHeight: '1.6', color: '#e5e7eb' }}>{parseInline(line)}</p>;
  });
};

const parseInline = (text) => {
  const parts = text.split(/\*\*([^*]+)\*\*/g);
  return parts.map((part, i) => {
    if (i % 2 === 1) {
      return <strong key={i} style={{ color: '#34d399', fontWeight: '600' }}>{part}</strong>;
    }
    const codeParts = part.split(/`([^`]+)`/g);
    return codeParts.map((subPart, j) => {
      if (j % 2 === 1) {
        return <code key={j} style={{ backgroundColor: '#1f2937', color: '#f472b6', padding: '2px 4px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '13px' }}>{subPart}</code>;
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
  // Authentication State
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return sessionStorage.getItem('lexior_auth') === 'true';
  });
  const [passcode, setPasscode] = useState('');
  const [authError, setAuthError] = useState(false);
  const [isShaking, setIsShaking] = useState(false);

  // Connection Config State
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || 'https://6eys2nzfy3u10a-8000.proxy.runpod.net/v1');
  const [apiKey, setApiKey] = useState(import.meta.env.VITE_API_KEY || 'none');
  const [modelId, setModelId] = useState(import.meta.env.VITE_MODEL_ID || 'intelliwork/LexiorGpt1-merged');
  const [systemPrompt, setSystemPrompt] = useState(
    'Tu es un assistant juridique Lexior, spécialisé en droit canadien et québécois. Raisonne en français. Tu dois obligatoirement baser tes analyses sur la législation et la jurisprudence canadienne/québécoise (ex: Code civil du Québec, CanLII).'
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

  // Handle Passcode Unlock
  const handleUnlock = (e) => {
    e.preventDefault();
    const correctPasscode = import.meta.env.VITE_APP_PASSWORD || 'Lexior2026';
    if (passcode === correctPasscode) {
      sessionStorage.setItem('lexior_auth', 'true');
      setIsAuthenticated(true);
      setAuthError(false);
    } else {
      setAuthError(true);
      setIsShaking(true);
      setPasscode('');
      setTimeout(() => setIsShaking(false), 400);
    }
  };

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
    } else {
      setActiveTab('chat');
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

    // Si la question utilisateur est courte, on renforce l'ancrage français dans l'historique
    const preparedMessages = [
      { role: 'system', content: systemPrompt }
    ];

    // Ancrage français pour les prompts courts (< 50 caractères) pour bloquer la dérive pinyin
    if (promptText.length < 50) {
      preparedMessages.push({ 
        role: 'system', 
        content: "CRITICAL REMINDER: You must reply ONLY in French. Do not use any Chinese or English characters." 
      });
    }

    preparedMessages.push(...newMessages);

    try {
      const response = await fetch(`${apiUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: modelId,
          messages: preparedMessages,
          temperature: temperature,
          max_tokens: maxTokens,
          repetition_penalty: 1.05, // Décourager les répétitions et dérives de jetons
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

  // --- RENDER SECURE GATE ---
  if (!isAuthenticated) {
    return (
      <div className="auth-overlay">
        <div className={`glass-panel auth-card ${isShaking ? 'auth-shake' : ''}`}>
          <img src="/logo.png" className="auth-logo" alt="Logo LexiorGPT" />
          <h2 className="auth-title">LexiorGPT Console</h2>
          <p className="auth-subtitle">Veuillez entrer la clé d'accès pour déverrouiller la console d'évaluation.</p>
          
          <form onSubmit={handleUnlock} style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div className="form-group" style={{ textAlign: 'left' }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Lock style={{ width: '12px', height: '12px' }} /> Clé d'accès
              </label>
              <input 
                type="password" 
                className="glass-input" 
                placeholder="••••••••"
                style={{ textAlign: 'center', fontSize: '16px', letterSpacing: '0.1em' }}
                value={passcode}
                onChange={(e) => setPasscode(e.target.value)}
              />
            </div>
            
            {authError && (
              <span style={{ color: '#ef4444', fontSize: '11px', fontWeight: '500' }}>
                Clé d'accès incorrecte. Veuillez réessayer.
              </span>
            )}
            
            <button type="submit" className="glass-button primary" style={{ height: '44px', width: '100%', fontWeight: '600' }}>
              <Unlock style={{ width: '16px', height: '16px' }} /> Déverrouiller la Session
            </button>
          </form>
        </div>
      </div>
    );
  }

  // --- RENDER MAIN APPLICATION ---
  return (
    <div className="app-container">
      {/* 1. LEFT COLUMN: Configuration Sidebar */}
      <aside className="sidebar-panel">
        <div className="sidebar-header">
          <img src="/logo.png" className="sidebar-logo" alt="Logo LexiorGPT" />
          <h1 style={{ fontSize: '18px', fontWeight: 'bold', margin: 0, color: 'white' }}>LexiorGPT Console</h1>
        </div>

        {/* API connection parameters */}
        <div className="sidebar-section">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <h3 className="sidebar-section-title" style={{ flex: 1, margin: 0 }}>
              <Settings style={{ width: '14px', height: '14px', marginRight: '6px', verticalAlign: 'middle' }} /> Paramètres d'API
            </h3>
            <span style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              display: 'inline-block',
              backgroundColor: connectionStatus === 'ready' ? '#10b981' :
                               connectionStatus === 'testing' ? '#6366f1' :
                               connectionStatus === 'error' ? '#ef4444' : '#6b7280',
              boxShadow: connectionStatus === 'ready' ? '0 0 10px #10b981' : 
                         connectionStatus === 'error' ? '0 0 10px #ef4444' : 'none'
            }}></span>
          </div>

          <div className="form-group">
            <label className="form-label">Endpoint URL (RunPod vLLM)</label>
            <input 
              type="text" 
              className="glass-input" 
              style={{ fontSize: '12px' }}
              value={apiUrl} 
              onChange={(e) => setApiUrl(e.target.value)} 
            />
          </div>

          <div className="form-group">
            <label className="form-label">Hugging Face Token</label>
            <input 
              type="password" 
              className="glass-input" 
              style={{ fontSize: '12px' }}
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)} 
            />
          </div>

          <div className="form-group">
            <label className="form-label">Model ID</label>
            <input 
              type="text" 
              className="glass-input" 
              style={{ fontSize: '12px' }}
              value={modelId} 
              onChange={(e) => setModelId(e.target.value)} 
            />
          </div>

          <button onClick={testConnection} className="glass-button" style={{ fontSize: '12px', width: '100%' }}>
            <RefreshCw style={{ width: '14px', height: '14px' }} /> Tester la Connexion
          </button>
        </div>

        {/* Hyperparameters Config */}
        <div className="sidebar-section">
          <h3 className="sidebar-section-title">Configuration Inférence</h3>
          
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
              <span style={{ color: '#9ca3af' }}>Température</span>
              <span style={{ color: '#10b981', fontWeight: 'bold' }}>{temperature}</span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              style={{ accentColor: '#10b981', cursor: 'pointer' }}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
              <span style={{ color: '#9ca3af' }}>Max Tokens</span>
              <span style={{ color: '#6366f1', fontWeight: 'bold' }}>{maxTokens}</span>
            </div>
            <input 
              type="range" 
              min="100" 
              max="4096" 
              step="50"
              style={{ accentColor: '#6366f1', cursor: 'pointer' }}
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            />
          </div>
        </div>

        {/* System Prompt Config */}
        <div className="sidebar-section" style={{ flex: 1, borderBottom: 'none' }}>
          <label className="form-label" style={{ fontWeight: '600', marginBottom: '8px' }}>System Prompt (Instructions)</label>
          <textarea 
            className="glass-input" 
            style={{ flex: 1, resize: 'none', fontSize: '12px', lineHeight: '1.5' }} 
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>
      </aside>

      {/* 2. RIGHT COLUMN: Interactive Workspaces */}
      <main className="main-workspace">
        {/* Top Header Navigation Tabs */}
        <header className="workspace-header">
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              onClick={() => setActiveTab('chat')} 
              className={`glass-button ${activeTab === 'chat' ? 'active' : ''}`}
            >
              <Activity style={{ width: '16px', height: '16px' }} /> Playground & Performance
            </button>
            <button 
              onClick={() => setActiveTab('bench')} 
              className={`glass-button ${activeTab === 'bench' ? 'active' : ''}`}
            >
              <FileText style={{ width: '16px', height: '16px' }} /> Benchmark d'Évaluation CoT
            </button>
            <button 
              onClick={() => setActiveTab('doc')} 
              className={`glass-button ${activeTab === 'doc' ? 'active' : ''}`}
            >
              <BookOpen style={{ width: '16px', height: '16px' }} /> Méthode & Dataset
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '12px', color: '#9ca3af' }}>Note Session :</span>
            <span style={{
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              color: '#10b981',
              fontWeight: 'bold',
              padding: '4px 12px',
              borderRadius: '9999px',
              border: '1px solid rgba(16, 185, 129, 0.2)',
              fontSize: '12px'
            }}>{getAverageScore()}</span>
          </div>
        </header>

        {/* WORKSPACE 1: CHAT & PERFORMANCE WORKSPACE */}
        {activeTab === 'chat' && (
          <div className="workspace-body">
            {/* Chat Feed Column (60%) */}
            <div className="chat-column">
              <div className="chat-messages-area">
                {messages.map((msg, i) => (
                  <div 
                    key={i} 
                    className={`message-bubble ${msg.role === 'user' ? 'user' : 'assistant'}`}
                  >
                    <span className="message-bubble-header">
                      {msg.role === 'user' ? 'Utilisateur' : 'LexiorGPT (vLLM)'}
                    </span>
                    <div className="message-text-content">
                      {formatMarkdown(msg.content)}
                    </div>
                  </div>
                ))}
                
                {isStreaming && (
                  <div className="message-bubble assistant">
                    <span className="message-bubble-header">LexiorGPT (vLLM)</span>
                    <div className="message-text-content" style={{ display: 'flex', gap: '4px', padding: '14px 18px', width: 'fit-content' }}>
                      <div className="dot"></div>
                      <div className="dot"></div>
                      <div className="dot"></div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Bar */}
              <div className="chat-input-bar">
                <input 
                  type="text" 
                  className="glass-input" 
                  style={{ flex: 1, height: '44px', fontSize: '13px' }}
                  placeholder="Posez votre question juridique (Ex: Règle d'indemnité de préavis...)" 
                  value={currentInput}
                  onChange={(e) => setCurrentInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  disabled={isStreaming}
                />
                <button 
                  onClick={() => handleSend()} 
                  className="glass-button primary" 
                  style={{ height: '44px', width: '50px' }}
                  disabled={isStreaming}
                >
                  <Send style={{ width: '16px', height: '16px' }} />
                </button>
              </div>
            </div>

            {/* Performance Panel Column (40%) */}
            <div className="telemetry-column">
              <h3 style={{ fontSize: '13px', fontWeight: 'bold', color: '#9ca3af', textTransform: 'uppercase', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity style={{ width: '16px', height: '16px', color: '#6366f1' }} /> Télémétrie en Temps Réel
              </h3>

              {/* Grid Widgets */}
              <div className="telemetry-grid">
                <div className="glass-panel widget-card">
                  <span className="widget-title">Latence Premier Token</span>
                  <span className="widget-value">{latency > 0 ? `${latency} ms` : '-'}</span>
                  <span className="widget-subtext" style={{ color: '#6366f1' }}>Calculé sur l'API vLLM</span>
                </div>

                <div className="glass-panel widget-card">
                  <span className="widget-title">Vitesse Inférence</span>
                  <span className="widget-value">{tokensPerSec > 0 ? `${tokensPerSec} t/s` : '-'}</span>
                  <span className="widget-subtext" style={{ color: '#10b981' }}>Vitesse de génération</span>
                </div>
              </div>

              {/* Token Usage Widget */}
              <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span className="widget-title">Consommation des Tokens</span>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
                  <span style={{ color: '#9ca3af' }}>Tokens d'Entrée (Prompt) :</span>
                  <span style={{ color: 'white', fontWeight: '500' }}>{inputTokens}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '6px' }}>
                  <span style={{ color: '#9ca3af' }}>Tokens de Sortie (Générés) :</span>
                  <span style={{ color: 'white', fontWeight: '500' }}>{outputTokens}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', paddingTop: '4px' }}>
                  <span style={{ color: '#e5e7eb', fontWeight: '600' }}>Total :</span>
                  <span style={{ color: '#6366f1', fontWeight: 'bold' }}>{inputTokens + outputTokens}</span>
                </div>
              </div>

              {/* Performance SVG Line Chart */}
              <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <span className="widget-title">Historique de Vitesse (tokens/sec)</span>
                <div style={{ width: '100%', height: '120px', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', display: 'flex', alignItems: 'end', padding: '8px', position: 'relative', overflow: 'hidden' }}>
                  <svg style={{ width: '100%', height: '100%' }} viewBox="0 0 100 50" preserveAspectRatio="none">
                    <polyline
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2"
                      points={speedHistory.map((val, idx) => {
                        const x = (idx / (speedHistory.length - 1)) * 100;
                        const y = 50 - (val / 100) * 50; 
                        return `${x},${y}`;
                      }).join(' ')}
                    />
                  </svg>
                  <div style={{ position: 'absolute', right: '8px', top: '8px', backgroundColor: 'rgba(99,102,241,0.1)', color: '#6366f1', fontSize: '9px', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(99,102,241,0.2)' }}>
                    Max: 100 t/s
                  </div>
                </div>
              </div>

              {/* Hardware diagnostics */}
              <div className="glass-panel" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <AlertCircle style={{ width: '20px', height: '20px', color: '#6366f1', minWidth: '20px' }} />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <span style={{ fontSize: '12px', fontWeight: '600', color: 'white' }}>Modèle Qwen-32B non quantifié</span>
                  <span style={{ fontSize: '10px', color: '#9ca3af', lineHeight: '1.4' }}>
                    Exécution en FP16 natif sur A100. Vitesse et cache d'attention flash (FlashAttention-2) optimisés.
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 2: BENCHMARK EVALUATION WORKSPACE */}
        {activeTab === 'bench' && (
          <div className="benchmark-layout">
            {/* Scenarios Panel (30%) */}
            <aside className="scenarios-sidebar">
              <div style={{ padding: '16px', borderBottom: '1px solid var(--glass-border)' }}>
                <h3 style={{ fontSize: '12px', fontWeight: 'bold', color: '#9ca3af', uppercase: 'true', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BookOpen style={{ width: '16px', height: '16px', color: '#10b981' }} /> Scénarios de Test
                </h3>
              </div>
              <div className="scenarios-list">
                {scenarios.map((sc, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedScenarioIdx(idx)}
                    className={`scenario-button ${selectedScenarioIdx === idx ? 'active' : ''}`}
                  >
                    <h4 style={{ fontSize: '12px', fontWeight: 'bold', margin: 0, color: selectedScenarioIdx === idx ? '#10b981' : 'white' }}>
                      {sc.title}
                    </h4>
                    <p style={{ fontSize: '10px', color: '#9ca3af', marginTop: '6px', margin: 0, lineHeight: '1.4' }}>
                      {sc.description}
                    </p>
                  </button>
                ))}
              </div>
            </aside>

            {/* Run & Verification Workspace (70%) */}
            <div className="evaluation-workspace">
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px' }}>
                  <div>
                    <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: 'white', margin: 0 }}>{scenarios[selectedScenarioIdx].title}</h2>
                    <p style={{ fontSize: '12px', color: '#9ca3af', margin: '4px 0 0 0' }}>{scenarios[selectedScenarioIdx].description}</p>
                  </div>
                  <button 
                    onClick={() => handleSend(scenarios[selectedScenarioIdx].prompt)}
                    className="glass-button primary" 
                    style={{ whiteSpace: 'nowrap' }}
                    disabled={isStreaming}
                  >
                    <Play style={{ width: '14px', height: '14px' }} /> Lancer l'Évaluation
                  </button>
                </div>
                
                <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: '10px', color: '#10b981', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Prompt envoyé :</span>
                  <p style={{ fontSize: '12px', color: '#d1d5db', margin: '6px 0 0 0', lineHeight: '1.5' }}>{scenarios[selectedScenarioIdx].prompt}</p>
                </div>
              </div>

              {/* Grid for criteria checklist & grading */}
              <div className="evaluation-grid">
                
                {/* Criteria checklist (3 columns) */}
                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: 'bold', color: 'white', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CheckCircle style={{ width: '16px', height: '16px', color: '#10b981' }} /> Critères de CoT Attendus
                  </h3>
                  <div className="checklist-container">
                    {scenarios[selectedScenarioIdx].criteria.map((crit, cIdx) => (
                      <div 
                        key={cIdx} 
                        onClick={() => toggleCriterion(selectedScenarioIdx, cIdx)}
                        className={`checklist-item ${crit.met ? 'checked' : ''}`}
                      >
                        <input 
                          type="checkbox" 
                          checked={crit.met} 
                          readOnly
                          style={{ marginTop: '3px', accentColor: '#10b981', cursor: 'pointer' }}
                        />
                        <span style={{ fontSize: '12px', lineHeight: '1.5' }}>{crit.text}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Performance score grading (2 columns) */}
                <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: 'bold', color: 'white', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Star style={{ width: '16px', height: '16px', color: '#f59e0b' }} /> Évaluation Qualitative
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

                  <div style={{ backgroundColor: 'rgba(245,158,11,0.03)', border: '1px solid rgba(245,158,11,0.1)', padding: '12px', borderRadius: '8px', display: 'flex', alignItems: 'start', gap: '10px', marginTop: '8px' }}>
                    <AlertCircle style={{ width: '16px', height: '16px', color: '#f59e0b', minWidth: '16px', marginTop: '2px' }} />
                    <span style={{ fontSize: '10px', color: '#f59e0b', lineHeight: '1.4' }}>
                      L'évaluation qualitative permet de mesurer la pertinence des explications étape par étape et l'alignement sur les politiques de l'organisation.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE 3: METHODOLOGY & DATASET DOCUMENTATION */}
        {activeTab === 'doc' && (
          <div className="evaluation-workspace" style={{ maxWidth: '1000px', margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Header Title Card */}
            <div className="glass-panel" style={{ padding: '24px', background: 'linear-gradient(135deg, rgba(99,102,241,0.05), rgba(16,185,129,0.05))', border: '1px solid rgba(255,255,255,0.08)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <img src="/logo.png" style={{ width: '64px', height: '64px', borderRadius: '14px', border: '1px solid rgba(245,158,11,0.3)' }} alt="Logo" />
                <div>
                  <h2 style={{ fontSize: '22px', fontWeight: 'bold', color: 'white', margin: 0 }}>Rapport technique : Distillation de LexiorGPT-1</h2>
                  <p style={{ fontSize: '13px', color: '#9ca3af', margin: '4px 0 0 0' }}>Comprendre la méthodologie d'entraînement, le jeu de données et l'infrastructure de service.</p>
                </div>
              </div>
            </div>

            {/* Content grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              {/* Left Column: Distillation details */}
              <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexSpace: 'column', flexDirection: 'column', gap: '16px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 'bold', color: '#34d399', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles style={{ width: '18px', height: '18px' }} /> Méthode de Distillation
                </h3>
                
                <p style={{ fontSize: '13px', color: '#d1d5db', lineHeight: '1.6', margin: 0 }}>
                  LexiorGPT-1 est un modèle distillé à partir de **DeepSeek-R1** (modèle enseignant) vers **Qwen-2.5-32B-Instruct** (modèle étudiant) à l'aide de la méthode de Fine-Tuning de précision **LoRA (Low-Rank Adaptation)**.
                </p>

                <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ fontSize: '11px', color: '#818cf8', fontWeight: 'bold' }}>Hyperparamètres d'entraînement :</span>
                  <ul style={{ margin: '8px 0 0 0', paddingLeft: '16px', fontSize: '12px', color: '#9ca3af', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <li><strong>Rank LoRA (r)</strong> : 16</li>
                    <li><strong>Alpha LoRA</strong> : 32</li>
                    <li><strong>Modules ciblés</strong> : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</li>
                    <li><strong>Époques</strong> : 3</li>
                    <li><strong>Taille de batch</strong> : 16 (avec accumulation de gradient)</li>
                    <li><strong>Loss finale</strong> : ~0.05 (convergence optimale)</li>
                  </ul>
                </div>

                <p style={{ fontSize: '12px', color: '#9ca3af', lineHeight: '1.5', margin: 0 }}>
                  La distillation LoRA permet de transférer la structure de raisonnement **Chain of Thought (CoT)** et la rigueur d'analyse logique de DeepSeek-R1 tout en conservant l'efficacité multilingue et le format d'instruction natif de Qwen-2.5.
                </p>
              </div>

              {/* Right Column: Dataset Structure */}
              <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 'bold', color: '#818cf8', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BookOpen style={{ width: '18px', height: '18px' }} /> Jeu de données (Dataset)
                </h3>

                <p style={{ fontSize: '13px', color: '#d1d5db', lineHeight: '1.6', margin: 0 }}>
                  Le modèle a été spécialisé sur un corpus de 293 cas d'entraînement ciblant les **Lois et Règlements Fédéraux du Canada** extraits de CanLII et de Justice Canada.
                </p>

                <div style={{ backgroundColor: 'rgba(0,0,0,0.15)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ fontSize: '11px', color: '#34d399', fontWeight: 'bold' }}>Structure et méthodologie du Dataset :</span>
                  <ul style={{ margin: '8px 0 0 0', paddingLeft: '16px', fontSize: '12px', color: '#9ca3af', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <li><strong>Lois fédérales majeures</strong> : Loi sur les produits dangereux, Rentes sur l'État, Protection civile, Hydrocarbures, Système correctionnel.</li>
                    <li><strong>Formalisme IRAC obligatoire</strong> : Structuration explicite en Issue (Question), Rule (Règle), Application (Analyse), Conclusion.</li>
                    <li><strong>Citations structurées</strong> : Références de bas de page formater sous forme de métadonnées JSON liant vers les lois CanLII.</li>
                  </ul>
                </div>

                <p style={{ fontSize: '12px', color: '#9ca3af', lineHeight: '1.5', margin: 0 }}>
                  Chaque exemple du dataset force le modèle à produire un bloc <code>&lt;think&gt;</code> contenant ses étapes de raisonnement, de citation d'articles et d'hypothèses juridiques avant d'apporter sa conclusion.
                </p>
              </div>
            </div>

            {/* Bottom Panel: serving infrastructure */}
            <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 'bold', color: 'white', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity style={{ width: '18px', height: '18px', color: '#6366f1' }} /> Moteur d'Inférence vLLM
              </h3>
              <p style={{ fontSize: '13px', color: '#d1d5db', lineHeight: '1.6', margin: 0 }}>
                Pour garantir une vitesse d'exécution de production compatible avec les playbooks SOAR, LexiorGPT-1 est servi via le moteur **vLLM (v0.25.1)**. 
                Ce moteur utilise des mécanismes avancés de gestion de la mémoire comme **PagedAttention** et le noyau **FlashAttention-2** pour exécuter le modèle en précision FP16 native sans goulot d'étranglement de VRAM, assurant un débit d'inférence moyen de 50+ tokens/seconde par utilisateur.
              </p>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
        <span style={{ color: '#9ca3af' }}>{label}</span>
        <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>{value} / 5</span>
      </div>
      <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => onChange(star)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '2px',
              color: star <= value ? '#f59e0b' : '#4b5563',
              transition: 'all 0.15s ease-in-out'
            }}
          >
            <Star style={{ width: '18px', height: '18px', fill: star <= value ? 'currentColor' : 'none' }} />
          </button>
        ))}
      </div>
    </div>
  );
}
