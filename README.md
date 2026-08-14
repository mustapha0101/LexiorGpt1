# LexiorGPT — live demo

Cette copie du projet est dédiée uniquement à la démonstration interactive du
système multi-agent Lexior.

```text
Phase1_Data_Preparation/      application live
docs/mcp_tools_catalog.json  catalogue des outils juridiques
.mcp.json                    serveurs MCP utilisés par l’application
.env                         clés et configuration locale, non versionnées
.venv/                       environnement Python local
```

## Démarrage

API :

```powershell
cd .\Phase1_Data_Preparation
..\.venv\Scripts\python.exe -m uvicorn lexior.api.app:app --port 8000 --reload
```

Interface, dans un deuxième terminal :

```powershell
cd .\Phase1_Data_Preparation\apps\chat-web
npm install
npm run dev
```

Ouvrir `http://127.0.0.1:5173`.

La documentation du parcours et des composants conservés se trouve dans
[`Phase1_Data_Preparation/README.md`](Phase1_Data_Preparation/README.md).
