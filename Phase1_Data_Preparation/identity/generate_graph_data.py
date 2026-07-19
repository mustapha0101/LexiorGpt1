#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération d'exemples d'entraînement pour la création d'ontologies 
et l'extraction de graphes de connaissances en droit canadien.
Permet d'entraîner le modèle à sortir des formats JSON bruts parfaits requis 
par le moteur d'Insights de LexiorNotebook.
"""

import os
import json

# --- 1. Scénarios pour la génération d'ontologies ---
ONTOLOGY_SAMPLES = [
    # Scénario A : Litige de construction / Immobilier
    {
        "text": "Bail de location commerciale conclu entre Construction Québec Inc. et Boutique Chic S.E.N.C. concernant le local 101. L'avenant signé le 14 mai par Pierre Roy (administrateur) prévoit des travaux d'insonorisation sous la supervision de l'expert technique Jean Dupont. Des fissures majeures sont apparues dans les fondations en raison d'infiltrations d'eau.",
        "ontology": {
            "entity_types": [
                {"name": "Contract", "description": "Legal agreement between parties (e.g. lease, amendment)", "attributes": [{"name": "title", "type": "text", "description": "Title of the contract"}, {"name": "sign_date", "type": "text", "description": "Signing date"}], "examples": ["Bail commercial", "Avenant"]},
                {"name": "Company", "description": "Incorporated business entity", "attributes": [{"name": "legal_name", "type": "text", "description": "Official company name"}], "examples": ["Construction Québec Inc."]},
                {"name": "Location", "description": "Physical property or premises", "attributes": [{"name": "address", "type": "text", "description": "Address of the premises"}], "examples": ["Local 101"]},
                {"name": "Expert", "description": "Technical expert or inspector", "attributes": [{"name": "specialty", "type": "text", "description": "Expert's domain of expertise"}], "examples": ["Jean Dupont"]},
                {"name": "Damage", "description": "Physical damage or defects identified", "attributes": [{"name": "type", "type": "text", "description": "Type of defect or damage"}], "examples": ["Fissures majeures"]},
                {"name": "Event", "description": "Key milestone or occurrence in the timeline", "attributes": [{"name": "date", "type": "text", "description": "Date of event"}], "examples": ["Signature du bail"]},
                {"name": "Lawsuit", "description": "Legal court proceedings", "attributes": [{"name": "district", "type": "text", "description": "Judicial district"}], "examples": ["Demande en dommages"]},
                {"name": "Clause", "description": "Specific clause of the contract", "attributes": [{"name": "number", "type": "text", "description": "Clause number"}], "examples": ["Clause d'insonorisation"]},
                {"name": "Person", "description": "Natural person fallback", "attributes": [{"name": "full_name", "type": "text", "description": "Person's name"}, {"name": "role", "type": "text", "description": "Role in dossier"}], "examples": ["Pierre Roy"]},
                {"name": "Organization", "description": "Any company, group, or institution fallback", "attributes": [{"name": "name", "type": "text", "description": "Organization name"}], "examples": ["Boutique Chic S.E.N.C."]}
            ],
            "edge_types": [
                {"name": "SIGNED_BY", "description": "A person signs a contract", "source_targets": [{"source": "Person", "target": "Contract"}], "attributes": []},
                {"name": "CONCERNS_PROPERTY", "description": "Contract concerns a location", "source_targets": [{"source": "Contract", "target": "Location"}], "attributes": []},
                {"name": "SUPERVISES", "description": "Expert supervises work or contract", "source_targets": [{"source": "Expert", "target": "Contract"}], "attributes": []},
                {"name": "AFFECTED_BY", "description": "Location is affected by damages", "source_targets": [{"source": "Location", "target": "Damage"}], "attributes": []},
                {"name": "PARTNER_OF", "description": "Relationship between two organizations", "source_targets": [{"source": "Organization", "target": "Company"}], "attributes": []}
            ]
        }
    },
    # Scénario B : Droit du travail / Congédiement
    {
        "text": "Contrat d'emploi à durée indéterminée de Sophie Lavoie au sein de Technologie Canada. Le licenciement a été notifié le 12 janvier par le directeur des ressources humaines Jacques Martin. L'employée conteste et dépose une plainte pour congédiement déguisé devant le Tribunal administratif du travail (TAT).",
        "ontology": {
            "entity_types": [
                {"name": "EmploymentContract", "description": "Labour or employment agreement", "attributes": [{"name": "start_date", "type": "text", "description": "Start date of employment"}], "examples": ["Contrat à durée indéterminée"]},
                {"name": "Employer", "description": "Company employing individuals", "attributes": [{"name": "name", "type": "text", "description": "Employer name"}], "examples": ["Technologie Canada"]},
                {"name": "Employee", "description": "The employed person", "attributes": [{"name": "job_title", "type": "text", "description": "Title of position"}], "examples": ["Sophie Lavoie"]},
                {"name": "TerminationNotice", "description": "Formal notice of termination of employment", "attributes": [{"name": "notice_date", "type": "text", "description": "Date of notice"}], "examples": ["Notification du 12 janvier"]},
                {"name": "Complaint", "description": "Legal action or dispute file", "attributes": [{"name": "ground", "type": "text", "description": "Grounds for complaint"}], "examples": ["Plainte pour congédiement déguisé"]},
                {"name": "Tribunal", "description": "Administrative tribunal or court", "attributes": [{"name": "name", "type": "text", "description": "Tribunal name"}], "examples": ["Tribunal administratif du travail"]},
                {"name": "HRDirector", "description": "Human resources representative", "attributes": [{"name": "name", "type": "text", "description": "Name of director"}], "examples": ["Jacques Martin"]},
                {"name": "CompensationClaim", "description": "Claim for financial compensation or indemnity", "attributes": [{"name": "amount", "type": "text", "description": "Indemnity amount claimed"}], "examples": ["Indemnité de préavis"]},
                {"name": "Person", "description": "Natural person fallback", "attributes": [{"name": "full_name", "type": "text", "description": "Person name"}], "examples": ["Sophie Lavoie"]},
                {"name": "Organization", "description": "Group or corporate entity fallback", "attributes": [{"name": "name", "type": "text", "description": "Organization name"}], "examples": ["Technologie Canada"]}
            ],
            "edge_types": [
                {"name": "EMPLOYED_BY", "description": "Employee works for employer", "source_targets": [{"source": "Employee", "target": "Employer"}], "attributes": []},
                {"name": "ISSUED_BY", "description": "Termination notice issued by HR Director", "source_targets": [{"source": "TerminationNotice", "target": "HRDirector"}], "attributes": []},
                {"name": "FILED_BEFORE", "description": "Complaint filed before a tribunal", "source_targets": [{"source": "Complaint", "target": "Tribunal"}], "attributes": []},
                {"name": "CONTESTS", "description": "Employee contests termination notice", "source_targets": [{"source": "Employee", "target": "TerminationNotice"}], "attributes": []}
            ]
        }
    }
]

# --- 2. Scénarios pour l'extraction de graphes ---
GRAPH_EXTRACTION_SAMPLES = [
    # Extraction pour le Scénario A (Construction)
    {
        "ontology": {
            "entity_types": ["Contract", "Company", "Location", "Expert", "Damage", "Person", "Organization"],
            "edge_types": ["SIGNED_BY", "CONCERNS_PROPERTY", "SUPERVISES", "AFFECTED_BY"]
        },
        "text": "L'avenant au bail commercial concernant le local 101 a été signé le 14 mai par Pierre Roy pour le compte de Boutique Chic S.E.N.C. L'expert technique Jean Dupont a été mandaté pour superviser les travaux. Des fissures majeures sont constatées sur place.",
        "graph": {
            "nodes": [
                {"id": "avenant_bail", "label": "Avenant au bail commercial", "type": "Contract", "description": "Avenant signé le 14 mai"},
                {"id": "local_101", "label": "Local 101", "type": "Location", "description": "Local commercial visé par le bail"},
                {"id": "pierre_roy", "label": "Pierre Roy", "type": "Person", "description": "Signataire représentant Boutique Chic"},
                {"id": "boutique_chic", "label": "Boutique Chic S.E.N.C.", "type": "Organization", "description": "Locataire du local 101"},
                {"id": "jean_dupont", "label": "Jean Dupont", "type": "Expert", "description": "Expert technique mandaté"},
                {"id": "fissures_majeures", "label": "Fissures majeures", "type": "Damage", "description": "Fissures constatées dans les fondations"}
            ],
            "edges": [
                {"source": "pierre_roy", "target": "avenant_bail", "relationship": "SIGNED_BY", "description": "Pierre Roy a signé l'avenant"},
                {"source": "avenant_bail", "target": "local_101", "relationship": "CONCERNS_PROPERTY", "description": "L'avenant concerne le local 101"},
                {"source": "jean_dupont", "target": "avenant_bail", "relationship": "SUPERVISES", "description": "Jean Dupont supervise les travaux de l'avenant"},
                {"source": "local_101", "target": "fissures_majeures", "relationship": "AFFECTED_BY", "description": "Le local 101 est affecté par des fissures"}
            ]
        }
    }
]

def main():
    output_file = "data/processed/generated_graph_cot.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        # 1. Génération des exemples SFT d'ontologies
        for sample in ONTOLOGY_SAMPLES:
            text = sample["text"]
            ontology_json = sample["ontology"]
            
            prompt = (
                "Analyze the following representative text from the workspace and design a custom ontology schema specifically suited to extract a knowledge graph from these documents.\n"
                "Return the ontology schema STRICTLY in JSON format matching this structure:\n"
                "{\n"
                "  \"entity_types\": [\n"
                "    {\n"
                "      \"name\": \"PascalCaseEntityTypeName\",\n"
                "      \"description\": \"Short description of the entity type (max 100 chars)\",\n"
                "      \"attributes\": [\n"
                "        {\n"
                "          \"name\": \"snake_case_attribute_name\",\n"
                "          \"type\": \"text\",\n"
                "          \"description\": \"Description of the attribute\"\n"
                "        }\n"
                "      ],\n"
                "      \"examples\": [\"example_1\", \"example_2\"]\n"
                "    }\n"
                "  ],\n"
                "  \"edge_types\": [\n"
                "    {\n"
                "      \"name\": \"UPPER_SNAKE_CASE_RELATION_NAME\",\n"
                "      \"description\": \"Short description of the relationship type (max 100 chars)\",\n"
                "      \"source_targets\": [\n"
                "        {\"source\": \"PascalCaseSourceType\", \"target\": \"PascalCaseTargetType\"}\n"
                "      ],\n"
                "      \"attributes\": []\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "CRITICAL RULES:\n"
                "1. You must design exactly 10 entity types.\n"
                "2. The last two entity types in the list MUST be the fallback types \"Person\" and \"Organization\".\n"
                "3. The first 8 entity types must be specific classes suited for the text.\n"
                "4. All entity types must represent concrete actors or entities. Do not use abstract concepts.\n"
                "5. Attributes MUST NOT use reserved names like \"name\", \"uuid\", \"id\".\n"
                "6. Return ONLY the raw JSON object. Do not wrap it in markdown code blocks.\n\n"
                f"TEXT SAMPLES:\n{text}"
            )
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(ontology_json, ensure_ascii=False)}
            ]
            
            # Répéter les exemples (facteur 40) pour bien ancrer le format JSON strict
            for _ in range(40):
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1

        # 2. Génération des exemples SFT d'extractions de graphes
        for sample in GRAPH_EXTRACTION_SAMPLES:
            ontology_schema_str = json.dumps(sample["ontology"], ensure_ascii=False, indent=2)
            allowed_entity_types = ", ".join([f'"{t}"' for t in sample["ontology"]["entity_types"]])
            allowed_edge_types = ", ".join([f'"{t}"' for t in sample["ontology"]["edge_types"]])
            text = sample["text"]
            graph_json = sample["graph"]
            
            prompt = (
                "You are an expert Named Entity Recognition (NER) pipeline and Graph Database modeler.\n"
                "Your task is to analyze the document text and extract entities and relationships strictly conforming to the target Ontology schema.\n\n"
                f"TARGET ONTOLOGY:\n{ontology_schema_str}\n\n"
                "RULES:\n"
                "- Return ONLY a valid JSON object matching the schema: {\"nodes\": [...], \"edges\": [...]}.\n"
                "- Conforming to this JSON structure:\n"
                "  {\n"
                "    \"nodes\": [\n"
                "      {\"id\": \"john_doe\", \"label\": \"John Doe\", \"type\": \"Person\", \"description\": \"Short description\"}\n"
                "    ],\n"
                "    \"edges\": [\n"
                "      {\"source\": \"john_doe\", \"target\": \"acme_corp\", \"relationship\": \"WORKS_FOR\", \"description\": \"John works at Acme Corp\"}\n"
                "    ]\n"
                "  }\n"
                "- Do NOT wrap your output in markdown blocks (no ```json).\n"
                "- For each node:\n"
                "  - \"id\": unique lowercase alphanumeric slug (e.g. \"john_doe\").\n"
                "  - \"label\": display name (e.g. \"John Doe\").\n"
                f"  - \"type\": MUST exactly match one of the permitted types: [{allowed_entity_types}].\n"
                "  - \"description\": A brief context description.\n"
                "- For each edge:\n"
                "  - \"source\": The source node id.\n"
                "  - \"target\": The target node id.\n"
                f"  - \"relationship\": MUST exactly match one of the permitted relationship types: [{allowed_edge_types}].\n"
                "  - \"description\": An explanation of the relationship.\n\n"
                f"DOCUMENT TEXT:\n{text}\n"
            )
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(graph_json, ensure_ascii=False)}
            ]
            
            # Répéter (facteur 40)
            for _ in range(40):
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"Dataset d'extraction de graphes et d'ontologies généré ! {count} exemples créés dans '{output_file}'.")

if __name__ == "__main__":
    main()
