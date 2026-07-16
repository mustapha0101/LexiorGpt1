#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de données d'entraînement pour les appels d'outils (Tool Calling).
Permet d'enseigner au modèle comment et quand appeler les outils MCP du projet LexiorNotebook
(recherche CCQ, CPC, jurisprudence CanLII, Éducaloi, Commission d'accès à l'information et Barreau).
"""

import os
import json

# Définition des outils au format JSON-Schema
TOOLS_DEFINITION = [
    {
        "name": "get_ccq_articles",
        "description": "Récupère le texte officiel d'un ou plusieurs articles du Code civil du Québec (CCQ).",
        "parameters": {
            "type": "object",
            "properties": {
                "start_article": {"type": "number", "description": "Le numéro de l'article de départ (ex: 1371)"},
                "end_article": {"type": "number", "description": "Le numéro de l'article de fin (ex: 1378). Facultatif."}
            },
            "required": ["start_article"]
        }
    },
    {
        "name": "search_ccq_keywords",
        "description": "Recherche par mots-clés dans les articles du CCQ. Utile pour trouver les numéros d'articles liés à un concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Le mot-clé à rechercher (ex: 'vice caché', 'louage')"}
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "search_quebec_jurisprudence",
        "description": "Recherche des décisions de justice (jurisprudence) québécoise sur CanLII et SOQUIJ.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La question de droit ou les mots-clés (ex: 'obligation de diligence médecin')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_educaloi",
        "description": "Recherche des articles de vulgarisation juridique sur Éducaloi (droit québécois).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le sujet de recherche (ex: 'reprise de logement par propriétaire')"},
                "language": {"type": "string", "enum": ["fr", "en"], "default": "fr"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_cai",
        "description": "Recherche dans les guides de conformité de la Loi 25 et décisions de la Commission d'accès à l'information (CAI) du Québec.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le sujet (ex: 'fuite de données personnelles Loi 25')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_barreau_deontology",
        "description": "Recherche dans le Code de déontologie des avocats du Québec et les directives de rigueur du Barreau.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le concept déontologique (ex: 'secret professionnel et conflit d'intérêts')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_knowledge_graph",
        "description": "Interroge le graphe de connaissances (Knowledge Graph) de LexiorNotebook pour trouver les relations, les liens d'évidence et les connexions indirectes entre différentes entités (personnes, organisations, lieux, documents ou lois).",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Les entités à analyser dans le graphe (ex: ['Jean Tremblay', 'Hydro-Québec'])"
                },
                "query": {
                    "type": "string",
                    "description": "Une question ou description optionnelle pour guider la recherche sémantique dans le graphe"
                }
            },
            "required": ["entities"]
        }
    }
]

# Modèles de conversations SFT avec appels d'outils
TOOL_CALLING_SAMPLES = [
    # 1. get_ccq_articles
    {
        "user_query": "Peux-tu me donner l'article 1457 du Code civil du Québec ?",
        "thinking_1": "L'utilisateur veut lire l'article 1457 du CCQ. Je dois appeler l'outil get_ccq_articles pour récupérer le texte exact afin d'éviter toute erreur de citation.",
        "tool_name": "get_ccq_articles",
        "tool_args": {"start_article": 1457},
        "tool_output": "Article 1457 CCQ: Toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, l'usage ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui...",
        "thinking_2": "J'ai obtenu le texte officiel de l'article 1457. Je vais le citer mot pour mot et l'expliquer brièvement comme demandé.",
        "final_assistant_response": "Voici le texte officiel de l'**article 1457 du Code civil du Québec** :\n\n> « Toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, l'usage ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui. »\n\nCet article est la base de la responsabilité civile extracontractuelle au Québec, exigeant la réunion de trois éléments : une faute, un préjudice et un lien de causalité."
    },
    # 2. search_ccq_keywords
    {
        "user_query": "Quel article du Code civil parle de la garantie contre les vices cachés ?",
        "thinking_1": "L'utilisateur s'interroge sur les vices cachés. Je dois rechercher dans le CCQ les articles contenant 'vice caché' à l'aide de l'outil search_ccq_keywords.",
        "tool_name": "search_ccq_keywords",
        "tool_args": {"keyword": "vice caché"},
        "tool_output": "Article 1726 CCQ: Le vendeur est tenu de garantir à l'acheteur que le bien et ses accessoires sont, lors de la vente, exempts de vices cachés qui le rendent impropre à l'usage auquel on le destine...",
        "thinking_2": "L'outil a retourné l'article 1726 du CCQ. Je vais répondre en citant cet article clé sur la garantie de qualité (vices cachés).",
        "final_assistant_response": "La garantie contre les vices cachés au Québec est régie par l'**article 1726 du Code civil du Québec**. \n\nCelui-ci dispose que :\n> « Le vendeur est tenu de garantir à l'acheteur que le bien et ses accessoires sont, lors de la vente, exempts de vices cachés qui le rendent impropre à l'usage auquel on le destine... »"
    },
    # 3. search_quebec_jurisprudence
    {
        "user_query": "Je cherche des jugements québécois sur la responsabilité civile en cas d'empiètement de clôture.",
        "thinking_1": "L'utilisateur veut de la jurisprudence sur l'empiètement de clôture au Québec. Je dois lancer une recherche sur CanLII/SOQUIJ à l'aide de search_quebec_jurisprudence.",
        "tool_name": "search_quebec_jurisprudence",
        "tool_args": {"query": "responsabilité civile empiètement clôture terrain"},
        "tool_output": "1. Tremblay c. Roy, 2021 QCCS 143: Conflit de voisinage concernant l'empiètement d'une clôture de 15 cm. Application de l'article 953 CCQ. 2. Gagnon c. Bureau, 2018 QCCQ 8943: Clôture mitoyenne et empiètement de mauvaise foi.",
        "thinking_2": "J'ai des décisions pertinentes (Tremblay c. Roy et Gagnon c. Bureau) faisant application de l'article 953 CCQ. Je vais résumer ces options jurisprudentielles pour l'utilisateur.",
        "final_assistant_response": "Voici deux décisions de jurisprudence pertinentes concernant l'empiètement de clôture au Québec :\n\n1. **Tremblay c. Roy (2021 QCCS 143)** : Ce jugement traite d'un empiètement mineur de 15 cm. La Cour supérieure a appliqué l'article 953 CCQ pour déterminer si la clôture devait être démolie ou si une indemnité était préférable.\n2. **Gagnon c. Bureau (2018 QCCQ 8943)** : Traite de la mauvaise foi de la partie ayant érigé la clôture mitoyenne sans consentement."
    },
    # 4. search_educaloi
    {
        "user_query": "Quels sont les droits d'un locataire face à une reprise de logement au Québec ?",
        "thinking_1": "C'est une question de droit du logement québécois. Je vais chercher un article de vulgarisation clair sur Éducaloi à l'aide de search_educaloi.",
        "tool_name": "search_educaloi",
        "tool_args": {"query": "reprise de logement par le propriétaire"},
        "tool_output": "Titre: La reprise de logement. Résumé: Au Québec, un propriétaire peut reprendre un logement pour s'y loger ou y loger ses proches, sous certaines conditions strictes et délais de préavis (6 mois avant la fin du bail de 12 mois).",
        "thinking_2": "J'ai les détails d'Éducaloi sur les conditions et les préavis de reprise de logement. Je vais synthétiser ces règles en citant la source.",
        "final_assistant_response": "Selon les ressources d'**Éducaloi**, au Québec, un propriétaire peut légalement reprendre un logement pour y habiter lui-même ou y loger ses proches parents, mais sous conditions :\n\n* **Préavis** : Le locateur doit envoyer un avis écrit 6 mois avant la fin du bail (pour un bail de 12 mois).\n* **Indemnité** : Le propriétaire peut devoir verser une indemnité pour les frais de déménagement."
    },
    # 5. search_cai
    {
        "user_query": "Qui doit-on nommer comme responsable de la protection des renseignements personnels au Québec avec la Loi 25 ?",
        "thinking_1": "La question porte sur la Loi 25 (protection des renseignements personnels) au Québec. Je dois interroger l'outil search_cai pour obtenir les directives de la Commission d'accès à l'information.",
        "tool_name": "search_cai",
        "tool_args": {"query": "responsable de la protection des renseignements personnels Loi 25"},
        "tool_output": "Loi 25 - CAI: Par défaut, la personne ayant la plus haute autorité au sein de l'entreprise (CEO/Directeur général) assume le rôle de responsable de la protection des renseignements personnels. Ce rôle peut être délégué par écrit.",
        "thinking_2": "La réponse de la CAI est claire : par défaut c'est la plus haute autorité de l'entreprise, avec possibilité de délégation écrite. Je formule la réponse finale.",
        "final_assistant_response": "Selon les règles de la **Loi 25** et les guides de la **Commission d'accès à l'information (CAI)** :\n\n* **Par défaut** : C'est la personne ayant la **plus haute autorité** au sein de votre organisation (dirigeant, président ou directeur général) qui assume ce rôle.\n* **Délégation** : Cette fonction peut être déléguée par écrit, en tout ou en partie, à un employé interne."
    },
    # 6. search_barreau_deontology
    {
        "user_query": "Quelles sont les obligations d'un avocat québécois face à un conflit d'intérêts potentiel ?",
        "thinking_1": "Il s'agit d'une question de déontologie des avocats au Québec. Je dois appeler search_barreau_deontology pour vérifier les obligations déontologiques officielles du Barreau du Québec.",
        "tool_name": "search_barreau_deontology",
        "tool_args": {"query": "conflit interets deontology"},
        "tool_output": "Code de déontologie des avocats (Québec), section conflit d'intérêts: L'avocat doit éviter toute situation de conflit d'intérêts. Il ne peut représenter deux clients ayant des intérêts divergents à moins d'avoir obtenu leur consentement écrit préalable et éclairé.",
        "thinking_2": "Le Code de déontologie impose d'éviter le conflit et requiert un consentement écrit éclairé si la double représentation est tolérée. Je réponds à l'utilisateur.",
        "final_assistant_response": "En vertu du **Code de déontologie des avocats du Québec**, un avocat a l'obligation stricte d'éviter toute situation de conflit d'intérêts :\n\n1. **Interdiction de principe** : L'avocat ne peut pas conseiller ou représenter des parties ayant des intérêts opposés.\n2. **Exception** : Une double représentation est possible uniquement si l'avocat obtient un **consentement écrit, libre et éclairé** de chacun des clients après leur avoir exposé la situation."
    },
    # 7. query_knowledge_graph
    {
        "user_query": "Existe-t-il des liens répertoriés dans nos documents entre l'entreprise A et le gestionnaire B ?",
        "thinking_1": "L'utilisateur veut découvrir des relations complexes ou indirectes entre deux entités (l'entreprise A et le gestionnaire B) à travers tous les documents classés. Je dois interroger le graphe de connaissances en utilisant query_knowledge_graph.",
        "tool_name": "query_knowledge_graph",
        "tool_args": {"entities": ["Entreprise A", "Gestionnaire B"]},
        "tool_output": "Chemin de connexion trouvé dans le graphe: [Entreprise A] -- (a signé le contrat 202) --> [Contrat 202] -- (approuvé par) --> [Gestionnaire B]. De plus, [Gestionnaire B] est cité comme administrateur de [Filiale C], détenue à 100% par [Entreprise A].",
        "thinking_2": "Le graphe de connaissances a trouvé deux chemins reliant les entités : une relation contractuelle directe et un poste d'administrateur dans une filiale commune. Je vais synthétiser ces liens d'évidence.",
        "final_assistant_response": "Oui, l'analyse du **graphe de connaissances (Knowledge Graph)** montre deux connexions distinctes entre l'**Entreprise A** et le **Gestionnaire B** :\n\n1. **Lien contractuel** : Le Gestionnaire B a approuvé et signé le *Contrat 202* liant les deux parties.\n2. **Lien corporatif indirect** : Le Gestionnaire B est inscrit comme administrateur de la *Filiale C*, une entité détenue à 100 % par l'Entreprise A."
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
