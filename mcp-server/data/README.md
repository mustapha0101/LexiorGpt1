# Base de données CCQ

Afin d'éviter d'être bloqué par les protections anti-bot (Imperva) du site officiel LégisQuébec, le serveur a besoin d'avoir accès au Code civil du Québec localement.

## Instructions
1. Ouvrez votre navigateur web (Google Chrome, Safari, etc.).
2. Allez sur cette page officielle : `https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991/xml`
3. Le navigateur va télécharger un fichier nommé `CCQ-1991.xml`.
4. Placez ce fichier dans ce dossier (`packages/backend/ccq-mcp/data/`).
5. Redémarrez le serveur MCP.

Le serveur MCP pourra ainsi lire instantanément des milliers d'articles sans jamais interroger le réseau et sans risque d'être banni.
