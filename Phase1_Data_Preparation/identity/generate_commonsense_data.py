#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de données d'entraînement pour les situations de droit civil de la vie quotidienne
(prêt d'objets, perte d'animaux confiés, vol/reprise unilatérale).
Évite les hallucinations absurdes du modèle (comme citer la loi sur les animaux de l'Alberta pour un vélo).
"""

import os
import json

SYSTEM_PROMPT = (
    "Tu es LexiorGPT, un assistant d'intelligence juridique spécialisé en droit canadien et québécois. "
    "Raisonne en français selon le format IRAC. Tu as été conçu et développé par l'équipe d'intelliwork."
)

COMMONSENSE_SAMPLES = [
    # 1. Joe et le vélo de Biden
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "joe a volé le vélo de biden que lui a emprunté a joe avant noel"},
            {
                "role": "assistant",
                "content": "<tool_call>{\"name\": \"ccq_keyword_search\", \"arguments\": {\"keyword\": \"prêt à usage restitution\"}}</tool_call>",
                "thinking": "L'utilisateur pose une question sur un prêt de vélo suivi d'une reprise unilatérale. Pour éviter toute hallucination sur les articles de loi et citer exactement les articles sur le prêt à usage (commodat) et le retrait du bien, je dois faire une recherche par mots-clés dans le CCQ."
            },
            {
                "role": "tool",
                "name": "ccq_keyword_search",
                "content": "Article 2313 CCQ: Le prêt à usage est un contrat à titre gratuit par lequel une personne, le prêteur, met un bien à la disposition d'une autre, l'emprunteur, à charge pour elle de le lui restituer...\nArticle 2320 CCQ: Le prêteur ne peut retirer le bien à l'emprunteur qu'après le terme convenu, ou, à défaut de terme, après qu'il a cessé d'en avoir besoin..."
            },
            {
                "role": "assistant",
                "content": (
                    "**Issue (Question de droit) :** Joe a-t-il le droit de reprendre unilatéralement le vélo qu'il avait prêté à Joe Biden avant Noël ? Biden a-t-il un recours pour réclamer le vélo ?\n\n"
                    "**Rule (Règle de droit applicable) :**\n"
                    "1. Selon l'**article 2313 du Code civil du Québec (CCQ)**, le prêt à usage (ou commodat) est un contrat par lequel le prêteur met un bien à la disposition de l'emprunteur, à charge pour lui de le lui restituer.\n"
                    "2. En vertu de l'**article 2320 du CCQ**, le prêteur ne peut pas retirer le bien à l'emprunteur avant le terme convenu, ou, à défaut de terme, avant qu'il ait cessé d'en avoir besoin, sauf si un besoin pressant et imprévu survient.\n"
                    "3. De plus, selon les principes généraux du droit de propriété et de la responsabilité civile (**article 1457 du CCQ**), nul ne peut se faire justice à soi-même (justice privée). Reprendre un bien par la force ou en cachette constitue une faute civile, même si l'on en est le propriétaire légitime.\n\n"
                    "**Application :** Joe avait prêté son vélo à Biden (prêt à usage). Même s'il en demeure le propriétaire, il n'avait pas le droit de le reprendre unilatéralement et sans avertissement (qualifié ici de « vol » par l'utilisateur). Si aucun terme n'était échu, Biden avait le droit de conserver la garde du vélo. En agissant par justice privée, Joe commet une faute civile.\n\n"
                    "**Conclusion :** Joe ne pouvait pas reprendre le vélo unilatéralement. Biden est en droit de réclamer la restitution du vélo pour la durée restante du prêt ou de demander des dommages-intérêts pour le préjudice causé par cette reprise fautive.\n\n"
                    "[^1]:{\"type\":\"url\",\"url\":\"https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991#se:2313\",\"title\":\"Code civil du Québec - Article 2313\"}"
                ),
                "thinking": "J'ai obtenu les articles 2313 et 2320 du CCQ officiels via la recherche. Je vais maintenant formuler le raisonnement IRAC de manière rigoureuse en français québécois."
            }
        ]
    },
    # 2. Prêt de tondeuse au voisin
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "J'ai prêté ma tondeuse à mon voisin et il l'a cassée en l'utilisant. Qui doit payer ?"},
            {
                "role": "assistant",
                "content": "<tool_call>{\"name\": \"ccq_keyword_search\", \"arguments\": {\"keyword\": \"emprunteur détérioration bien\"}}</tool_call>",
                "thinking": "L'utilisateur demande qui est responsable des dommages causés à un objet prêté. Je dois interroger la base CCQ pour trouver les articles exacts régissant les obligations et responsabilités de l'emprunteur (détérioration et usage)."
            },
            {
                "role": "tool",
                "name": "ccq_keyword_search",
                "content": "Article 2327 CCQ: L'emprunteur est tenu de veiller, en bon père de famille, à la garde et à la conservation du bien prêté...\nArticle 2328 CCQ: L'emprunteur ne répond pas des détériorations qui résultent de l'usage normal et sans faute du bien..."
            },
            {
                "role": "assistant",
                "content": (
                    "**Issue (Question de droit) :** Le voisin (emprunteur) est-il responsable des dommages causés à la tondeuse prêtée et tenu de rembourser le propriétaire ?\n\n"
                    "**Rule (Règle de droit applicable) :**\n"
                    "1. En vertu de l'**article 2327 du Code civil du Québec (CCQ)**, l'emprunteur est tenu de veiller à la garde et à la conservation du bien prêté avec la prudence et la diligence d'une personne raisonnable.\n"
                    "2. Selon l'**article 2326 du CCQ**, l'emprunteur doit utiliser le bien conformément à sa destination et le restituer à la fin du prêt.\n"
                    "3. L'**article 2328 du CCQ** précise que l'emprunteur ne répond pas des détériorations qui résultent de l'usage normal du bien sans faute de sa part. C'est le propriétaire qui assume les risques de perte par force majeure ou usure normale.\n\n"
                    "**Application :** Si votre voisin a commis une négligence ou une faute (ex. heurter volontairement des pierres, forcer le moteur de manière abusive), il est responsable et doit payer les réparations (articles 2326 et 2327 CCQ). Cependant, si la panne est survenue en raison de l'usure normale d'une pièce mécanique vétuste, le voisin n'a commis aucune faute et c'est vous (le propriétaire) qui devez en assumer le coût (article 2328 CCQ).\n\n"
                    "**Conclusion :** Le voisin doit payer les réparations uniquement s'il est démontré qu'il a causé le bris par une mauvaise utilisation ou un manque de soin. S'il s'agit d'une usure normale, le coût vous incombe.\n\n"
                    "[^1]:{\"type\":\"url\",\"url\":\"https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991#se:2327\",\"title\":\"Code civil du Québec - Article 2327\"}"
                ),
                "thinking": "J'ai les articles officiels du CCQ (2327 et 2328) concernant le prêt. Je formule le raisonnement IRAC précis."
            }
        ]
    },
    # 3. Chien gardé pendant les vacances
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Mon ami m'a laissé son chien pendant ses vacances. Le chien s'est sauvé de ma cour. Est-ce que je suis responsable légalement ?"},
            {
                "role": "assistant",
                "content": "<tool_call>{\"name\": \"ccq_keyword_search\", \"arguments\": {\"keyword\": \"dépôt garde soin bien\"}}</tool_call>",
                "thinking": "L'utilisateur interroge sur la responsabilité lors de la perte d'un animal confié. C'est un contrat de dépôt gratuit. Je dois interroger le CCQ pour trouver les articles applicables au soin et à la responsabilité du dépositaire."
            },
            {
                "role": "tool",
                "name": "ccq_keyword_search",
                "content": "Article 2280 CCQ: Le dépôt est le contrat par lequel une personne remet un bien à une autre... qui s'oblige à le garder...\nArticle 2283 CCQ: Le dépositaire doit apporter, dans la garde du bien, le même soin qu'il apporte à la garde de ses propres biens..."
            },
            {
                "role": "assistant",
                "content": (
                    "**Issue (Question de droit) :** Le gardien bénévole (dépositaire gratuit) d'un chien est-il responsable de sa perte s'il s'échappe de sa propriété ?\n\n"
                    "**Rule (Règle de droit applicable) :**\n"
                    "1. Selon l'**article 2280 du Code civil du Québec (CCQ)**, le dépôt est le contrat par lequel une personne remet un bien (ou un animal) à un dépositaire qui s'oblige à le garder et à le restituer.\n"
                    "2. En vertu de l'**article 2283 du CCQ**, le dépositaire doit apporter, dans la garde du bien, le même soin qu'il apporte à la garde de ses propres biens.\n"
                    "3. L'**article 2285 du CCQ** prévoit que le dépositaire ne répond pas de la perte du bien par force majeure ou par sa nature propre, à mais qu'il n'ait commis une faute.\n\n"
                    "**Application :** En acceptant de garder le chien de votre ami gratuitement, vous avez conclu un contrat de dépôt. Si le chien s'est sauvé parce que vous avez laissé la barrière ouverte par négligence ou manque de surveillance, vous avez manqué à votre obligation de garde (article 2283 CCQ) et êtes responsable civilement. En revanche, si le chien a creusé sous une clôture solide ou s'est enfui suite à un événement imprévisible (ex: foudre), vous n'avez pas commis de faute et n'êtes pas responsable.\n\n"
                    "**Conclusion :** Votre responsabilité légale dépend de votre diligence. Si la fuite est attribuable à une négligence de votre part, vous êtes responsable; sinon, aucune faute ne peut vous être reprochée.\n\n"
                    "[^1]:{\"type\":\"url\",\"url\":\"https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991#se:2283\",\"title\":\"Code civil du Québec - Article 2283\"}"
                ),
                "thinking": "Les articles 2280 et 2283 du CCQ ont été récupérés avec succès. Je rédige l'application pratique en français québécois."
            }
        ]
    }
]

def main():
    output_file = "data/processed/generated_commonsense_cot.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for sample in COMMONSENSE_SAMPLES:
            # Réplication forte (facteur 40) pour forcer le modèle à bien comprendre ces situations de bon sens
            # et à abandonner les lois farfelues
            for _ in range(40):
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"Dataset de situations quotidiennes généré ! {count} exemples créés dans '{output_file}'.")

if __name__ == "__main__":
    main()
