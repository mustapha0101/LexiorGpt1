import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import axios from 'axios';
import * as cheerio from 'cheerio';
import Exa from 'exa-js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { loadCcqDb, loadCpcDb, getArticles, searchArticlesByKeyword, getCpcArticles, searchCpcArticlesByKeyword } from './db';
import { searchQuebecRegulations, searchQuebecJurisprudence, scrapeQuebecLegalInfo } from './scraper';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Initialize Exa Client for direct fetches
const exaKey = process.env.EXA_API_KEY || process.env.COPILOT_EXA_KEY || '1c3736ac-d902-4089-9d5e-cbe3ef66b578';
const exa = new Exa(exaKey);

// We load both databases asynchronously before starting the server
let isDbLoaded = false;

// Initialize MCP Server
const mcpServer = new Server({
  name: "lexior-legisquebec-mcp",
  version: "1.0.0",
}, {
  capabilities: {
    tools: {},
  }
});

// Define tools
mcpServer.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_ccq_articles",
        description: "Récupère le texte officiel d'un ou plusieurs articles du Code civil du Québec (CCQ).",
        inputSchema: {
          type: "object",
          properties: {
            start_article: {
              type: "number",
              description: "Le numéro de l'article de départ (ex: 1371)"
            },
            end_article: {
              type: "number",
              description: "Le numéro de l'article de fin (ex: 1698). Si non fourni, seul l'article de départ est renvoyé."
            }
          },
          required: ["start_article"]
        }
      },
      {
        name: "search_ccq_keywords",
        description: "Recherche par mots-clés dans le texte des articles du Code civil du Québec (CCQ). Utile pour trouver les numéros d'articles liés à un concept (ex: contrats, location).",
        inputSchema: {
          type: "object",
          properties: {
            keyword: {
              type: "string",
              description: "Le mot-clé ou l'expression à rechercher (ex: 'louage', 'contrats entre entreprises')"
            }
          },
          required: ["keyword"]
        }
      },
      {
        name: "get_cpc_articles",
        description: "Récupère le texte officiel d'un ou plusieurs articles du Code de procédure civile du Québec (CPC / C-25.01).",
        inputSchema: {
          type: "object",
          properties: {
            start_article: {
              type: "number",
              description: "Le numéro de l'article de départ (ex: 1)"
            },
            end_article: {
              type: "number",
              description: "Le numéro de l'article de fin (ex: 50). Si non fourni, seul l'article de départ est renvoyé."
            }
          },
          required: ["start_article"]
        }
      },
      {
        name: "search_cpc_keywords",
        description: "Recherche par mots-clés dans le texte des articles du Code de procédure civile du Québec (CPC / C-25.01). Utile pour trouver les numéros d'articles liés à un concept de procédure (ex: 'outrage au tribunal', 'signification').",
        inputSchema: {
          type: "object",
          properties: {
            keyword: {
              type: "string",
              description: "Le mot-clé ou l'expression à rechercher (ex: 'outrage au tribunal', 'délai d'appel')"
            }
          },
          required: ["keyword"]
        }
      },
      {
        name: "search_quebec_regulations",
        description: "Recherche sémantique parmi les règlements du Québec sur LégisQuébec.",
        inputSchema: {
          type: "object",
          properties: {
            keyword: {
              type: "string",
              description: "Le sujet ou mot-clé de recherche de règlement (ex: 'fixation des pensions alimentaires')"
            }
          },
          required: ["keyword"]
        }
      },
      {
        name: "get_quebec_regulation",
        description: "Récupère le contenu complet d'un règlement spécifique ou d'un acte législatif sur LégisQuébec à partir de son URL.",
        inputSchema: {
          type: "object",
          properties: {
            url: {
              type: "string",
              description: "L'URL LégisQuébec du règlement (ex: 'https://www.legisquebec.gouv.qc.ca/fr/document/rc/C-25.01,%20r.%200.4')"
            }
          },
          required: ["url"]
        }
      },
      {
        name: "get_quebec_legal_info",
        description: "Récupère des métadonnées et tableaux de modifications, entrées en vigueur et dispositions non en vigueur de LégisQuébec.",
        inputSchema: {
          type: "object",
          properties: {
            type: {
              type: "string",
              enum: ["modlois", "modreg", "nevlois", "eevlois", "loisann", "regann", "decision"],
              description: "Le type de tableau à récupérer (ex: 'modlois' pour les modifications des lois, 'nevlois' pour les dispositions non en vigueur)"
            }
          },
          required: ["type"]
        }
      },
      {
        name: "search_quebec_jurisprudence",
        description: "Recherche de la jurisprudence (décisions de justice) québécoise sur CanLII et SOQUIJ.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "La question de droit ou les mots-clés de recherche de jurisprudence (ex: 'responsabilité du développeur de logiciel')"
            }
          },
          required: ["query"]
        }
      }
    ]
  };
});

mcpServer.setRequestHandler(CallToolRequestSchema, async (request) => {
  console.log('--- 📩 NOUVEL APPEL D\'OUTIL ---');
  console.log(`Outil demandé : ${request.params.name}`);
  console.log(`Arguments :`, request.params.arguments);

  if (!isDbLoaded) {
    console.warn('❌ Erreur : Les bases de données ne sont pas prêtes.');
    return {
      content: [{ type: "text", text: "Erreur : Les bases de données ne sont pas encore prêtes ou sont manquantes." }],
      isError: true,
    };
  }

  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "get_ccq_articles": {
        const { start_article, end_article } = args as any;
        if (typeof start_article !== 'number') {
          return { content: [{ type: "text", text: "Erreur : start_article doit être un nombre." }], isError: true };
        }
        const articles = getArticles(start_article, end_article);
        if (articles.length === 0) {
          return { content: [{ type: "text", text: `Aucun article trouvé pour la plage CCQ ${start_article} à ${end_article || start_article}.` }] };
        }
        const formattedText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        return { content: [{ type: "text", text: formattedText }] };
      }

      case "search_ccq_keywords": {
        const { keyword } = args as any;
        if (typeof keyword !== 'string' || !keyword) {
          return { content: [{ type: "text", text: "Erreur : keyword doit être une chaîne de caractères." }], isError: true };
        }
        const articles = searchArticlesByKeyword(keyword);
        if (articles.length === 0) {
          return { content: [{ type: "text", text: `Aucun article CCQ trouvé pour le mot-clé: "${keyword}".` }] };
        }
        const formattedText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        return { content: [{ type: "text", text: formattedText }] };
      }

      case "get_cpc_articles": {
        const { start_article, end_article } = args as any;
        if (typeof start_article !== 'number') {
          return { content: [{ type: "text", text: "Erreur : start_article doit être un nombre." }], isError: true };
        }
        const articles = getCpcArticles(start_article, end_article);
        if (articles.length === 0) {
          return { content: [{ type: "text", text: `Aucun article CPC trouvé pour la plage CPC ${start_article} à ${end_article || start_article}.` }] };
        }
        const formattedText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        return { content: [{ type: "text", text: formattedText }] };
      }

      case "search_cpc_keywords": {
        const { keyword } = args as any;
        if (typeof keyword !== 'string' || !keyword) {
          return { content: [{ type: "text", text: "Erreur : keyword doit être une chaîne de caractères." }], isError: true };
        }
        const articles = searchCpcArticlesByKeyword(keyword);
        if (articles.length === 0) {
          return { content: [{ type: "text", text: `Aucun article CPC trouvé pour le mot-clé: "${keyword}".` }] };
        }
        const formattedText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        return { content: [{ type: "text", text: formattedText }] };
      }

      case "search_quebec_regulations": {
        const { keyword } = args as any;
        if (typeof keyword !== 'string' || !keyword) {
          return { content: [{ type: "text", text: "Erreur : keyword doit être une chaîne." }], isError: true };
        }
        const text = await searchQuebecRegulations(keyword);
        return { content: [{ type: "text", text }] };
      }

      case "get_quebec_regulation": {
        const { url } = args as any;
        if (typeof url !== 'string' || !url) {
          return { content: [{ type: "text", text: "Erreur : url doit être une chaîne." }], isError: true };
        }
        const text = await fetchDocumentContent(url);
        return { content: [{ type: "text", text }] };
      }

      case "get_quebec_legal_info": {
        const { type } = args as any;
        if (typeof type !== 'string' || !type) {
          return { content: [{ type: "text", text: "Erreur : type doit être une chaîne." }], isError: true };
        }
        const text = await scrapeQuebecLegalInfo(type);
        return { content: [{ type: "text", text }] };
      }

      case "search_quebec_jurisprudence": {
        const { query } = args as any;
        if (typeof query !== 'string' || !query) {
          return { content: [{ type: "text", text: "Erreur : query doit être une chaîne." }], isError: true };
        }
        const text = await searchQuebecJurisprudence(query);
        return { content: [{ type: "text", text }] };
      }

      default:
        throw new Error(`Outil non trouvé: ${name}`);
    }
  } catch (err: any) {
    console.error(`--- [MCP Error] Outil ${name} ---`, err);
    return {
      isError: true,
      content: [{ type: "text", text: `Erreur d'exécution: ${err.message}` }]
    };
  }
});

/**
 * Clean fetch utility for regulation HTML pages using Exa Crawl / Axios fallback
 */
async function fetchDocumentContent(url: string): Promise<string> {
  console.log(`[Fetch Document] URL: ${url}`);
  
  // First attempt: Exa Crawl for clean extraction
  try {
    const contentsResult = await exa.getContents([url], {
      text: { maxCharacters: 50000 }
    });
    if (contentsResult.results && contentsResult.results.length > 0 && contentsResult.results[0].text) {
      return `# ${contentsResult.results[0].title || 'Document LégisQuébec'}\n\n${contentsResult.results[0].text}\n\n---\n**Source :** ${url}`;
    }
  } catch (e: any) {
    console.warn(`[Fetch Document] Échec Exa Crawl : ${e.message}. Tentative alternative via Axios...`);
  }

  // Second attempt: Axios + Cheerio clean-up
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
      },
      timeout: 15000
    });

    const $ = cheerio.load(response.data);
    const contentArea = $('#contentDiv, .card-body, main, body').first();
    contentArea.find('script, style, iframe, form, nav, .list-group, .navbar').remove();
    
    const text = contentArea.text()
      .replace(/\n\s*\n/g, '\n\n')
      .replace(/\s+/g, ' ')
      .trim();

    return `# Document\n\n${text}\n\n---\n**Source :** ${url}`;
  } catch (err: any) {
    console.error(`[Fetch Document] Erreur Axios :`, err.message);
    throw new Error(`Impossible de récupérer le contenu : ${err.message}`);
  }
}

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const isStdioMode = process.argv.includes('--stdio') || process.env.TRANSPORT === 'stdio';
if (isStdioMode) {
  console.log = console.error;
}

// SSE Transport Map
let transport: SSEServerTransport | null = null;

app.post('/api/call-tool', async (req, res) => {
  const { name, arguments: args } = req.body;
  console.log(`--- [HTTP API] Appel d'outil direct : ${name} ---`, args);
  
  if (!isDbLoaded) {
    res.status(503).json({ error: "Bases de données non chargées." });
    return;
  }

  try {
    let resultText = "";
    switch (name) {
      case "get_ccq_articles": {
        const { start_article, end_article } = args;
        if (typeof start_article !== 'number') {
          res.status(400).json({ error: "start_article doit être un nombre." });
          return;
        }
        const articles = getArticles(start_article, end_article);
        resultText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        break;
      }
      case "search_ccq_keywords": {
        const { keyword } = args;
        if (typeof keyword !== 'string' || !keyword) {
          res.status(400).json({ error: "keyword doit être une chaîne." });
          return;
        }
        const articles = searchArticlesByKeyword(keyword);
        resultText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        break;
      }
      case "get_cpc_articles": {
        const { start_article, end_article } = args;
        if (typeof start_article !== 'number') {
          res.status(400).json({ error: "start_article doit être un nombre." });
          return;
        }
        const articles = getCpcArticles(start_article, end_article);
        resultText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        break;
      }
      case "search_cpc_keywords": {
        const { keyword } = args;
        if (typeof keyword !== 'string' || !keyword) {
          res.status(400).json({ error: "keyword doit être une chaîne." });
          return;
        }
        const articles = searchCpcArticlesByKeyword(keyword);
        resultText = articles.map((a: any) => `Article ${a.numero}\n${a.texte}`).join('\n\n');
        break;
      }
      case "search_quebec_regulations": {
        const { keyword } = args;
        if (typeof keyword !== 'string' || !keyword) {
          res.status(400).json({ error: "keyword doit être une chaîne." });
          return;
        }
        resultText = await searchQuebecRegulations(keyword);
        break;
      }
      case "get_quebec_regulation": {
        const { url } = args;
        if (typeof url !== 'string' || !url) {
          res.status(400).json({ error: "url doit être une chaîne." });
          return;
        }
        resultText = await fetchDocumentContent(url);
        break;
      }
      case "get_quebec_legal_info": {
        const { type } = args;
        if (typeof type !== 'string' || !type) {
          res.status(400).json({ error: "type doit être une chaîne." });
          return;
        }
        resultText = await scrapeQuebecLegalInfo(type);
        break;
      }
      case "search_quebec_jurisprudence": {
        const { query } = args;
        if (typeof query !== 'string' || !query) {
          res.status(400).json({ error: "query doit être une chaîne." });
          return;
        }
        resultText = await searchQuebecJurisprudence(query);
        break;
      }
      default:
        res.status(404).json({ error: `Outil non trouvé: ${name}` });
        return;
    }
    res.json({ content: [{ type: "text", text: resultText }] });
  } catch (err: any) {
    console.error("Erreur HTTP API tool call:", err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/sse', async (req, res) => {
  console.log('🔗 Nouvelle connexion SSE entrante pour LegisQuebec MCP.');
  transport = new SSEServerTransport('/messages', res);
  await mcpServer.connect(transport);
});

app.post('/messages', async (req, res) => {
  if (!transport) {
    res.status(400).send('Aucune connexion SSE active.');
    return;
  }
  await transport.handlePostMessage(req, res);
});

// Start Server / Transport
Promise.all([loadCcqDb(), loadCpcDb()]).then(async () => {
  isDbLoaded = true;
  console.log('✅ Bases de données CCQ et CPC chargées avec succès.');

  if (isStdioMode) {
    console.error('🚀 Démarrage du serveur MCP LégisQuébec en mode STDIO...');
    const stdioTransport = new StdioServerTransport();
    await mcpServer.connect(stdioTransport);
    console.error('✅ Serveur MCP connecté via STDIO.');
  } else {
    const PORT = process.env.PORT || 3001;
    app.listen(PORT, () => {
      console.log(`🚀 Serveur MCP LégisQuébec démarré sur le port ${PORT}`);
      console.log(`🌐 URL de connexion MCP (SSE) : http://localhost:${PORT}/sse`);
    });
  }
}).catch(err => {
  console.error('❌ Erreur lors du chargement des bases de données:', err);
  process.exit(1);
});

