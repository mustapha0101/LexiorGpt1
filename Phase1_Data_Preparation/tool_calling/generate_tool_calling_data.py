#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de données d'entraînement pour les appels d'outils (Tool Calling).
Permet d'enseigner au modèle comment et quand appeler les outils MCP du projet LexiorNotebook
(recherche CCQ, CPC, jurisprudence CanLII, Éducaloi, Commission d'accès à l'information et Barreau).
"""

import os
import json

# Définition des outils réels du backend au format JSON-Schema
TOOLS_DEFINITION = [
    {
        "name": "ccq_search",
        "description": "Récupère le texte officiel d'un ou plusieurs articles du Code civil du Québec (CCQ).",
        "parameters": {
            "type": "object",
            "properties": {
                "start_article": {"type": "number", "description": "Le numéro de l'article de départ (ex: 1371)"},
                "end_article": {"type": "number", "description": "Le numéro de l'article de fin (ex: 1698). Si non fourni, seul l'article de départ est renvoyé."}
            },
            "required": ["start_article"]
        }
    },
    {
        "name": "ccq_keyword_search",
        "description": "Recherche par mots-clés dans le texte des articles du Code civil du Québec (CCQ). Utile pour trouver les numéros d'articles liés à un concept (ex: contrats, location).",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Le mot-clé ou l'expression à rechercher (ex: 'louage', 'contrats entre entreprises')"}
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "cpc_search",
        "description": "Récupère le texte officiel d'un ou plusieurs articles du Code de procédure civile du Québec (CPC / C-25.01).",
        "parameters": {
            "type": "object",
            "properties": {
                "start_article": {"type": "number", "description": "Le numéro de l'article de départ (ex: 1)"},
                "end_article": {"type": "number", "description": "Le numéro de l'article de fin (ex: 50). Si non fourni, seul l'article de départ est renvoyé."}
            },
            "required": ["start_article"]
        }
    },
    {
        "name": "cpc_keyword_search",
        "description": "Recherche par mots-clés dans le texte des articles du Code de procédure civile du Québec (CPC / C-25.01).",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Le mot-clé ou l'expression de procédure à rechercher (ex: 'outrage au tribunal', 'signification')"}
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "a2aj_search_legal_documents",
        "description": "Search Canadian case law and legislation. Use this for any Canadian legal research instead of web search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "MUST BE A KEYWORD SEARCH. DO NOT USE NATURAL LANGUAGE. Supports boolean operators (AND/OR/NOT), quotes, wildcards (*), proximity ('A B'~n)"},
                "search_type": {"type": "string", "enum": ["full_text", "name"], "default": "full_text", "description": "'full_text' searches document content with snippets, 'name' searches titles only"},
                "doc_type": {"type": "string", "enum": ["cases", "laws"], "default": "cases", "description": "'cases' for Canadian case law, 'laws' for statutes & regulations"},
                "size": {"type": "number", "description": "Number of results to return (max 50)"},
                "search_language": {"type": "string", "enum": ["en", "fr"], "default": "en", "description": "Language to search in - 'en' or 'fr'"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "a2aj_fetch_document",
        "description": "Retrieve full text of Canadian legal documents by citation (e.g., '2020 SCC 5', 'RSC 1985, c C-46').",
        "parameters": {
            "type": "object",
            "properties": {
                "citation": {"type": "string", "description": "Official legal citation (e.g., '2020 SCC 5' or 'RSC 1985, c C-46')"},
                "doc_type": {"type": "string", "enum": ["cases", "laws"], "default": "cases", "description": "'cases' for Canadian case law, 'laws' for statutes & regulations"},
                "output_language": {"type": "string", "enum": ["en", "fr", "both"], "default": "en", "description": "Language for output - 'en', 'fr', or 'both'"}
            },
            "required": ["citation"]
        }
    },
    {
        "name": "workspace_insights_query",
        "description": "Permet de requêter le graphe de connaissances (Insights) de l'espace de travail pour trouver des entités (Personnes, Organisations, Lois, Règlements, dates, etc.), leurs relations, et le contexte associé.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La question ou les mots-clés à rechercher dans le graphe de connaissances (ex: 'Jean Tremblay', 'liens entre ACME et contrat')."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "workspace_read_all",
        "description": "Extraire et lire l'intégralité des documents de l'espace de travail (workspace) actuel en une seule fois. À utiliser UNIQUEMENT lorsqu'une synthèse globale, un résumé complet ou 'faire le point sur l'ensemble des dossiers' est demandé. Ne pas utiliser pour une recherche de mots-clés spécifique.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "doc_read",
        "description": "Return the text content of a document (via Markdown) and its metadata. IMPORTANT: This tool DOES NOT extract text from attached files inside the document (like PDFs or Word files). If you need to analyze attachments, you MUST use the `doc_analyze_attachments` tool instead or in addition to this one.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "The target doc to read"}
            },
            "required": ["doc_id"]
        }
    },
    {
        "name": "doc_analyze_attachments",
        "description": "Extraire et analyser le contenu texte d'une pièce jointe spécifique (ex: PDF, DOCX, image) d'un document en utilisant un moteur d'extraction ou OCR.",
        "parameters": {
            "type": "object",
            "properties": {
                "attachment_id": {"type": "string", "description": "L'identifiant unique de la pièce jointe à analyser."},
                "query": {"type": "string", "description": "La question spécifique ou consigne d'analyse concernant la pièce jointe."}
            },
            "required": ["attachment_id"]
        }
    },
    {
        "name": "search_educaloi",
        "description": "Recherche des articles de vulgarisation juridique sur Éducaloi (droit québécois). Retourne des résumés clairs et des URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le sujet de recherche (ex: 'reprise de logement par le propriétaire')"},
                "language": {"type": "string", "enum": ["fr", "en"], "default": "fr"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_cai",
        "description": "Recherche dans les directives de conformité, guides sur la Loi 25 (protection des renseignements personnels) et décisions de la Commission d'accès à l'information (CAI) du Québec.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le sujet de conformité ou de décision CAI (ex: 'nomination responsable Loi 25')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_barreau_deontology",
        "description": "Recherche dans les règles déontologiques, le Code de déontologie des avocats et les directives de rigueur professionnelle du Barreau du Québec.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le concept déontologique recherché (ex: 'secret professionnel et conflit d'intérêts')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "studio_actions",
        "description": "Permet de lancer des analyses approfondies en arrière-plan via le module Studio (actions standards: 'flow_chart' / 'flowchart' / 'logigramme' pour générer un logigramme ou flowchart, 'mindmap' pour une carte mentale, 'legal_timeline' pour une chronologie, 'presentation' pour une présentation, 'validate_contract', 'analyze_risks', 'global_synthesis', 'contradiction_detector', 'import_and_classify'). L'action sera traitée de manière asynchrone.",
        "parameters": {
            "type": "object",
            "properties": {
                "actionType": {"type": "string", "description": "L'identifiant de l'action à lancer. Exemples standard : 'flow_chart', 'mindmap', 'legal_timeline', 'presentation', 'global_synthesis', 'validate_contract', 'contradiction_detector', 'lexior_skill:calc_delais_cpc'."},
                "scope": {"type": "string", "enum": ["document", "workspace"], "default": "workspace", "description": "L'étendue de l'analyse : 'document' pour le document actuel, 'workspace' pour l'ensemble du dossier."}
            },
            "required": ["actionType"]
        }
    }
]

# Modèles de conversations SFT avec appels d'outils
TOOL_CALLING_SAMPLES = [
    # 1. ccq_search
    {
        "user_query": "Peux-tu me citer l'article 1457 du Code civil du Québec ?",
        "thinking_1": "L'utilisateur demande le texte de l'article 1457 du CCQ. Je dois appeler l'outil ccq_search pour obtenir la version officielle et éviter toute hallucination.",
        "tool_name": "ccq_search",
        "tool_args": {"start_article": 1457},
        "tool_output": "Article 1457 CCQ: Toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, l'usage ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui. Elle est tenue, lorsqu'elle manque à ce devoir, de réparer le préjudice causé à autrui par sa faute, qu'elle soit de nature corporelle, morale ou matérielle. Elle est aussi tenue, en certains cas, de réparer le préjudice causé à autrui par le fait ou la faute d'une autre personne ou par le fait des biens qu'elle a sous sa garde.\n\nSource officielle: https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991#se:1457",
        "thinking_2": "J'ai obtenu le texte officiel de l'article 1457. Je vais le citer mot pour mot et l'expliquer brièvement comme demandé.",
        "final_assistant_response": "Voici le texte officiel de l'**article 1457 du Code civil du Québec** :\n\n> « Toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, l'usage ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui.\n> \n> Elle est tenue, lorsqu'elle manque à ce devoir, de réparer le préjudice causé à autrui par sa faute, qu'elle soit de nature corporelle, morale ou matérielle... »\n\nCet article est la base de la responsabilité civile extracontractuelle au Québec, exigeant la réunion de trois éléments : une faute, un préjudice et un lien de causalité."
    },
    # 2. ccq_keyword_search
    {
        "user_query": "Quels sont les articles du CCQ qui traitent des vices cachés dans une vente ?",
        "thinking_1": "L'utilisateur pose une question sur un concept juridique précis ('vice caché'). Je dois chercher les articles pertinents dans le CCQ à l'aide de l'outil ccq_keyword_search.",
        "tool_name": "ccq_keyword_search",
        "tool_args": {"keyword": "vice caché"},
        "tool_output": "Article 1726 CCQ: Le vendeur est tenu de garantir à l'acheteur que le bien et ses accessoires sont, lors de la vente, exempts de vices cachés qui le rendent impropre à l'usage auquel on le destine ou qui en diminuent tellement l'utilité que l'acheteur ne l'aurait pas acheté...",
        "thinking_2": "L'outil a retourné l'article 1726 comme principal résultat pour 'vice caché'. Je vais l'expliquer dans le contexte d'une vente.",
        "final_assistant_response": "En droit québécois, la garantie de qualité (vices cachés) est principalement régie par l'**article 1726 du Code civil du Québec** :\n\n> « Le vendeur est tenu de garantir à l'acheteur que le bien et ses accessoires sont, lors de la vente, exempts de vices cachés qui le rendent impropre à l'usage auquel on le destine... »"
    },
    # 3. cpc_search
    {
        "user_query": "Affiche-moi l'article 49 du Code de procédure civile du Québec.",
        "thinking_1": "L'utilisateur demande le texte d'un article spécifique du CPC (article 49). Je dois appeler cpc_search pour obtenir la rédaction exacte.",
        "tool_name": "cpc_search",
        "tool_args": {"start_article": 49},
        "tool_output": "Article 49 CPC: Les tribunaux du premier degré relevant de la compétence du Québec sont la Cour du Québec, la Cour supérieure et la Cour d'appel.\n\nSource officielle: https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-25.01#se:49",
        "thinking_2": "J'ai obtenu la définition officielle des tribunaux de premier degré au Québec selon l'article 49 du CPC. Je la transmets à l'utilisateur.",
        "final_assistant_response": "Voici le texte de l'**article 49 du Code de procédure civile du Québec** :\n\n> « Les tribunaux du premier degré relevant de la compétence du Québec sont la Cour du Québec, la Cour supérieure et la Cour d'appel. »"
    },
    # 4. cpc_keyword_search
    {
        "user_query": "Où est définie la signification par huissier dans le CPC ?",
        "thinking_1": "L'utilisateur cherche les règles ou définitions de la signification par huissier dans la procédure civile. Je lance une recherche par mot-clé à l'aide de cpc_keyword_search.",
        "tool_name": "cpc_keyword_search",
        "tool_args": {"keyword": "signification"},
        "tool_output": "Article 109 CPC: La notification par huissier s'opère par la remise d'une copie du document par un huissier de justice... Article 110 CPC: L'huissier dresse un procès-verbal de signification...",
        "thinking_2": "La signification par huissier est principalement couverte par les articles 109 et suivants du CPC. Je prépare le résumé.",
        "final_assistant_response": "Dans le **Code de procédure civile (CPC)**, la notification par huissier de justice est traitée aux **articles 109 et suivants** :\n\n* **Article 109 CPC** : Décrit les modalités de remise de la copie du document par l'huissier.\n* **Article 110 CPC** : Concerne le procès-verbal de signification qui sert de preuve."
    },
    # 5. a2aj_search_legal_documents
    {
        "user_query": "Trouve-moi des décisions récentes sur le congédiement déguisé au Québec.",
        "thinking_1": "L'utilisateur veut de la jurisprudence sur le congédiement déguisé. Je vais utiliser l'outil de recherche de documents juridiques A2AJ en filtrant sur la langue française et en ciblant la jurisprudence.",
        "tool_name": "a2aj_search_legal_documents",
        "tool_args": {"query": "\"congédiement déguisé\"", "doc_type": "cases", "search_language": "fr", "size": 5},
        "tool_output": "1. Potter c. Commission des services scolaires, 2015 CSC 10 (Source officielle: https://www.canlii.org/...)\nRésumé: La Cour suprême analyse les critères du congédiement déguisé...",
        "thinking_2": "L'outil a retourné l'arrêt Potter. Je vais structurer la réponse avec la méthodologie IRAC et inclure l'URL exacte reçue pour la citation.",
        "final_assistant_response": "La décision de référence concernant le congédiement déguisé au Canada est **Potter c. Commission des services scolaires (2015 CSC 10)**[^1]. La Cour suprême y détaille les deux approches pour établir le congédiement déguisé.\n\n[^1]: {\"type\":\"url\",\"url\":\"https://www.canlii.org/...\"}"
    },
    # 6. a2aj_fetch_document
    {
        "user_query": "Donne-moi les faits de la décision 2015 CSC 10.",
        "thinking_1": "L'utilisateur demande des détails sur une décision spécifique dont il fournit la citation (2015 CSC 10). Je dois exécuter a2aj_fetch_document pour analyser le texte complet.",
        "tool_name": "a2aj_fetch_document",
        "tool_args": {"citation": "2015 CSC 10", "doc_type": "cases", "output_language": "fr"},
        "tool_output": "[Texte du jugement Potter c. Commission des services scolaires...] M. Potter était directeur général de la commission scolaire. Suspendu avec solde par décision unilatérale...",
        "thinking_2": "J'ai extrait les faits à partir du texte officiel du jugement retourné. Je vais maintenant résumer ces faits de façon claire.",
        "final_assistant_response": "Dans l'affaire **Potter c. Commission des services scolaires (2015 CSC 10)**, les faits essentiels sont les suivants : M. Potter a été suspendu unilatéralement avec solde par la commission scolaire pendant la négociation d'une entente de départ..."
    },
    # 7. workspace_insights_query
    {
        "user_query": "Est-ce qu'il y a un lien documenté dans mon dossier entre Jean Tremblay et Hydro-Québec ?",
        "thinking_1": "L'utilisateur s'interroge sur les relations entre deux entités spécifiques dans son dossier. Je dois interroger le graphe de connaissances pour extraire les entités connectées et leurs relations.",
        "tool_name": "workspace_insights_query",
        "tool_args": {"query": "Jean Tremblay Hydro-Québec"},
        "tool_output": "{\"matchedNodes\": [{\"id\": \"node1\", \"label\": \"Jean Tremblay\", \"type\": \"Person\"}, {\"id\": \"node2\", \"label\": \"Hydro-Québec\", \"type\": \"Organization\"}], \"relationships\": [{\"source\": \"node1\", \"target\": \"node2\", \"type\": \"EMPLOYED_BY\", \"description\": \"Contrat de consultation signé le 12 mai 2024\"}]}",
        "thinking_2": "L'outil du graphe a renvoyé une relation de type EMPLOYED_BY. Je vais restituer cette information avec les détails du contrat de consultation.",
        "final_assistant_response": "Oui, selon le graphe de connaissances de votre espace de travail, **Jean Tremblay** a un lien d'emploi ou de consultation avec **Hydro-Québec** (contrat de consultation signé le 12 mai 2024)."
    },
    # 8. workspace_read_all
    {
        "user_query": "Peux-tu me faire un résumé global de toutes les notes de mon dossier actuel ?",
        "thinking_1": "L'utilisateur demande un résumé global de l'ensemble des documents de son espace de travail. Je dois utiliser l'outil workspace_read_all pour extraire l'intégralité du texte du workspace.",
        "tool_name": "workspace_read_all",
        "tool_args": {},
        "tool_output": "--- Document: Faits Chronologiques ---\nLe 12 Janvier 2025: Sinistre constaté...\n\n--- Document: Analyse Légale ---\nRisque de responsabilité civile selon 1457 CCQ...",
        "thinking_2": "J'ai obtenu le contenu global du workspace. Je vais maintenant en faire une synthèse ordonnée et claire.",
        "final_assistant_response": "Voici une synthèse globale de vos documents de travail :\n\n1. **Faits Chronologiques** : Un sinistre s'est produit le 12 janvier 2025...\n2. **Analyse Légale** : La responsabilité civile est envisagée sous l'angle de l'article 1457 du CCQ."
    },
    # 9. doc_read
    {
        "user_query": "Lis le document doc-sinistre-123 et donne-moi la date du sinistre.",
        "thinking_1": "L'utilisateur veut des informations spécifiques contenues dans le document doc-sinistre-123. Je dois appeler l'outil doc_read.",
        "tool_name": "doc_read",
        "tool_args": {"doc_id": "doc-sinistre-123"},
        "tool_output": "### Rapport de Sinistre\nLe sinistre des eaux a eu lieu le **14 mars 2025** dans les bureaux de l'entreprise...",
        "thinking_2": "J'ai lu le document et trouvé la date (14 mars 2025). Je réponds directement à la question.",
        "final_assistant_response": "D'après le document *Rapport de Sinistre*, le sinistre a eu lieu le **14 mars 2025**."
    },
    # 10. search_educaloi
    {
        "user_query": "Quels sont les droits d'un locataire face à une reprise de logement au Québec ?",
        "thinking_1": "C'est une question de droit du logement québécois. Je vais chercher un article de vulgarisation clair sur Éducaloi à l'aide de search_educaloi.",
        "tool_name": "search_educaloi",
        "tool_args": {"query": "reprise de logement par le propriétaire"},
        "tool_output": "Titre: La reprise de logement. Résumé: Au Québec, un propriétaire peut reprendre un logement pour s'y loger ou y loger ses proches, sous certaines conditions strictes et délais de préavis (6 mois avant la fin du bail de 12 mois).",
        "thinking_2": "J'ai les détails d'Éducaloi sur les conditions et les préavis de reprise de logement. Je vais synthétiser ces règles en citant la source.",
        "final_assistant_response": "Selon les ressources d'**Éducaloi**, au Québec, un propriétaire peut reprendre un logement pour y habiter lui-même ou y loger ses proches parents sous certaines conditions (avis écrit 6 mois avant la fin d'un bail de 12 mois)."
    },
    # 11. search_cai
    {
        "user_query": "Qui doit-on nommer comme responsable de la protection des renseignements personnels au Québec avec la Loi 25 ?",
        "thinking_1": "La question porte sur la Loi 25 (protection des renseignements personnels) au Québec. Je dois interroger l'outil search_cai pour obtenir les directives de la Commission d'accès à l'information.",
        "tool_name": "search_cai",
        "tool_args": {"query": "responsable de la protection des renseignements personnels Loi 25"},
        "tool_output": "Loi 25 - CAI: Par défaut, la personne ayant la plus haute autorité au sein de l'entreprise (CEO/Directeur général) assume le rôle de responsable de la protection des renseignements personnels. Ce rôle peut être délégué par écrit.",
        "thinking_2": "La réponse de la CAI est claire : par défaut c'est la plus haute autorité de l'entreprise, avec possibilité de délégation écrite. Je formule la réponse finale.",
        "final_assistant_response": "Selon les guides de la **Commission d'accès à l'information (CAI)**, sous la **Loi 25**, c'est par défaut la personne ayant la plus haute autorité au sein de l'organisation qui assume le rôle de responsable."
    },
    # 12. search_barreau_deontology
    {
        "user_query": "Quelles sont les obligations d'un avocat québécois face à un conflit d'intérêts potentiel ?",
        "thinking_1": "Il s'agit d'une question de déontologie des avocats au Québec. Je dois appeler search_barreau_deontology pour vérifier les obligations déontologiques officielles du Barreau du Québec.",
        "tool_name": "search_barreau_deontology",
        "tool_args": {"query": "conflit interets deontology"},
        "tool_output": "Code de déontologie des avocats (Québec), section conflit d'intérêts: L'avocat doit éviter toute situation de conflit d'intérêts. Il ne peut représenter deux clients ayant des intérêts divergents à moins d'avoir obtenu leur consentement écrit préalable et éclairé.",
        "thinking_2": "Le Code de déontologie impose d'éviter le conflit et requiert un consentement écrit éclairé si la double représentation est tolérée. Je réponds à l'utilisateur.",
        "final_assistant_response": "En vertu du **Code de déontologie des avocats du Québec**, un avocat doit éviter toute situation de conflit d'intérêts. Si une double représentation survient, elle exige un consentement écrit, libre et éclairé de chaque client."
    },
    # 13. ccq_keyword_search (Trouble de voisinage - Recherche MCP obligatoire)
    {
        "user_query": "Quelle est la règle légale concernant le trouble de voisinage au Québec ?",
        "thinking_1": "L'utilisateur pose une question sur les troubles de voisinage au Québec. Je dois impérativement chercher l'article de loi exact dans le CCQ via l'outil ccq_keyword_search au lieu de tenter de le citer de mémoire.",
        "tool_name": "ccq_keyword_search",
        "tool_args": {"keyword": "trouble de voisinage"},
        "tool_output": "Article 976 CCQ: Les voisins doivent subir les inconvénients normaux du voisinage qui ne dépassent pas la mesure de la tolérance qu'ils se doivent, suivant l'usage ou la condition des lieux.",
        "thinking_2": "J'ai récupéré l'article 976 du CCQ officiel. Je vais maintenant l'exposer à l'utilisateur sous la forme d'un raisonnement juridique structuré.",
        "final_assistant_response": "Au Québec, les troubles de voisinage sont régis par l'**article 976 du Code civil du Québec** :\n\n> « Les voisins doivent subir les inconvénients normaux du voisinage qui ne dépassent pas la mesure de la tolérance qu'ils se doivent, suivant l'usage ou la condition des lieux. »\n\nCette règle pose le principe d'une responsabilité sans faute : si un inconvénient dépasse la limite de la tolérance normale (selon le bruit, les odeurs ou l'usage des lieux), le voisin en est responsable même s'il n'a commis aucune faute délibérée."
    },
    # 14. doc_analyze_attachments (Analyse de pièce jointe)
    {
        "user_query": "Peux-tu analyser la pièce jointe 'att-rapport-expertise' et me donner les conclusions de l'expert ?",
        "thinking_1": "L'utilisateur me demande d'analyser le contenu d'une pièce jointe ('att-rapport-expertise'). Pour extraire et analyser le texte d'une pièce jointe (comme un PDF ou un document Word annexé), je dois appeler l'outil doc_analyze_attachments.",
        "tool_name": "doc_analyze_attachments",
        "tool_args": {"attachment_id": "att-rapport-expertise", "query": "Quelles sont les conclusions de l'expert ?"},
        "tool_output": "Conclusions du rapport d'expertise technique : Les fissures sur la façade ouest sont dues à un affaissement des fondations suite à une infiltration d'eau souterraine causée par un drain français obstrué.",
        "thinking_2": "L'outil d'analyse a extrait les conclusions du rapport d'expertise. Je vais en faire la synthèse claire pour l'utilisateur.",
        "final_assistant_response": "D'après l'analyse de la pièce jointe *att-rapport-expertise*, l'expert conclut que les fissures de la façade ouest résultent d'un affaissement des fondations provoqué par une infiltration d'eau due à l'obstruction du drain français."
    },
    # 15. studio_actions (Deadline calculator)
    {
        "user_query": "Lance le calculateur de délais CPC pour ce dossier.",
        "thinking_1": "L'utilisateur me demande de lancer le calculateur de délais de procédure civile (CPC) pour ce dossier. C'est une compétence du Lexior Store / Playbook. Je dois appeler l'outil studio_actions avec actionType='lexior_skill:calc_delais_cpc'.",
        "tool_name": "studio_actions",
        "tool_args": {"actionType": "lexior_skill:calc_delais_cpc", "scope": "workspace"},
        "tool_output": "{\"success\":true,\"message\":\"Calculateur de délais CPC lancé en tâche de fond pour l'espace de travail.\"}",
        "thinking_2": "L'action studio a été lancée avec succès en tâche de fond. Je réponds à l'utilisateur.",
        "final_assistant_response": "J'ai lancé le **Calculateur de Délais CPC** pour votre espace de travail. L'analyse des délais de procédure civile se poursuit en tâche de fond."
    },
    # 16. studio_actions (Contradiction detector)
    {
        "user_query": "Peux-tu lancer l'analyse de contradictions sur le document actuel ?",
        "thinking_1": "L'utilisateur demande de lancer l'analyse de contradictions sur le document actuel. C'est une action standard du Studio. Je dois appeler l'outil studio_actions avec actionType='contradiction_detector' et scope='document'.",
        "tool_name": "studio_actions",
        "tool_args": {"actionType": "contradiction_detector", "scope": "document"},
        "tool_output": "{\"success\":true,\"message\":\"Analyse de contradictions lancée pour le document actuel.\"}",
        "thinking_2": "L'action standard de détection de contradictions a été lancée avec succès pour le document. J'en informe l'utilisateur.",
        "final_assistant_response": "L'analyse de détection des contradictions a été lancée pour le document actuel. Les résultats s'afficheront directement dans le Studio sous peu."
    },
    # 17. workspace_insights_query (Knowledge Graph)
    {
        "user_query": "Montre-moi les connexions et relations dans mon graphe pour Hydro-Québec.",
        "thinking_1": "L'utilisateur demande les relations ou connexions du graphe de connaissances relatives à 'Hydro-Québec'. Je dois interroger le graphe via l'outil workspace_insights_query.",
        "tool_name": "workspace_insights_query",
        "tool_args": {"query": "Hydro-Québec"},
        "tool_output": "{\"nodes\": [{\"id\": \"node1\", \"label\": \"Hydro-Québec\"}], \"relationships\": [{\"source\": \"node1\", \"target\": \"node2\", \"type\": \"PARTNER\"}]}",
        "thinking_2": "J'ai obtenu les relations du graphe. Je vais les résumer clairement.",
        "final_assistant_response": "Voici les connexions trouvées dans le graphe pour **Hydro-Québec** : elle possède une relation de partenariat (PARTNER) avec l'entité liée."
    },
    # 18. workspace_read_all (Global synthesis)
    {
        "user_query": "Fais-moi un bilan global de mes dossiers et de tous les documents.",
        "thinking_1": "L'utilisateur demande une synthèse globale ou de faire le point sur tous les documents. Je dois lire l'intégralité du workspace en appelant workspace_read_all.",
        "tool_name": "workspace_read_all",
        "tool_args": {},
        "tool_output": "[Doc 1: Contrat de bail] Bail signé le 1er mai...\n[Doc 2: Mise en demeure] Envoyée le 10 juin...",
        "thinking_2": "J'ai lu tous les documents du workspace. Je fais le bilan global.",
        "final_assistant_response": "Voici le bilan global de vos documents : vous avez un contrat de bail signé le 1er mai et une mise en demeure envoyée le 10 juin."
    }
]

def main():
    output_file = "data/processed/generated_tool_calling_cot.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Prompt système intégrant les outils au format JSON
    system_prompt = (
        "Tu es LexiorGPT, un assistant d'intelligence juridique spécialisé en droit canadien et québécois. "
        "Tu as accès aux outils suivants pour valider tes citations et effectuer des recherches :\n"
        f"{json.dumps(TOOLS_DEFINITION, ensure_ascii=False, indent=2)}\n\n"
        "Raisonne en français selon le format IRAC. Utilise <tool_call> pour appeler un outil en cas de besoin."
    )
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for sample in TOOL_CALLING_SAMPLES:
            # Structurer la conversation avec les rôles appropriés
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["user_query"]},
                {
                    "role": "assistant",
                    "content": f"<tool_call>{json.dumps({'name': sample['tool_name'], 'arguments': sample['tool_args']}, ensure_ascii=False)}</tool_call>",
                    "thinking": sample["thinking_1"]
                },
                {
                    "role": "tool",
                    "name": sample["tool_name"],
                    "content": sample["tool_output"]
                },
                {
                    "role": "assistant",
                    "content": sample["final_assistant_response"],
                    "thinking": sample["thinking_2"]
                }
            ]
            
            message_data = {"messages": messages}
            
            # Répéter les exemples d'appels d'outils (facteur 20) pour forcer le modèle à bien comprendre
            # la syntaxe d'appel d'outil
            for _ in range(20):
                f_out.write(json.dumps(message_data, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"Dataset d'appels d'outils généré ! {count} exemples créés dans '{output_file}'.")

if __name__ == "__main__":
    main()
