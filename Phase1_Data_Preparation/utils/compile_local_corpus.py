#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de compilation et de structuration du corpus de lois locales.
Prend les fichiers ccqDb.json et cpcDb.json importés de LexiorNotebook et les organise
dans une arborescence de dossiers taxonomiques sous data/law_corpus/ pour un entraînement robuste.
"""

import os
import json

SOURCE_DIR = "data"
OUTPUT_DIR = "data/law_corpus"

def get_ccq_book(article_num):
    try:
        # Nettoyer les caractères non numériques
        clean_num = "".join([c for c in article_num if c.isdigit() or c in [".", ","]])
        num = float(clean_num.replace(",", "."))
    except ValueError:
        return "autres"
        
    if num <= 301:
        return "livre1_personnes"
    elif num <= 822:
        return "livre2_famille"
    elif num <= 946:
        return "livre3_successions"
    elif num <= 1370:
        return "livre4_biens"
    elif num <= 2643:
        return "livre5_obligations"
    elif num <= 2802:
        return "livre6_priorites_suretes"
    elif num <= 2874:
        return "livre7_preuve"
    elif num <= 2933:
        return "livre8_prescription"
    elif num <= 3023:
        return "livre9_publicite_droits"
    else:
        return "livre10_droit_international_prive"

def get_cpc_chapter(article_num):
    try:
        clean_num = "".join([c for c in article_num if c.isdigit() or c in [".", ","]])
        num = float(clean_num.replace(",", "."))
    except ValueError:
        return "autres"
        
    if num <= 81:
        return "titre1_dispositions_communes"
    elif num <= 140:
        return "titre2_modes_prives_reglement"
    elif num <= 320:
        return "titre3_procedure_contentieuse"
    elif num <= 535:
        return "titre4_jugement_recours"
    elif num <= 687:
        return "titre5_execution_forcee"
    else:
        return "titre6_procedures_non_contentieuses_et_autres"

def compile_corpus():
    print("Début de la structuration du corpus juridique local...")
    
    ccq_path = os.path.join(SOURCE_DIR, "ccqDb.json")
    cpc_path = os.path.join(SOURCE_DIR, "cpcDb.json")
    
    corpus_index = []
    
    # 1. Traitement du CCQ
    if os.path.exists(ccq_path):
        print(f"Ingestion du Code civil du Québec depuis {ccq_path}...")
        with open(ccq_path, "r", encoding="utf-8") as f:
            ccq_data = json.load(f)
            
        ccq_count = 0
        for item in ccq_data:
            num = item["numero"]
            text = item["texte"]
            
            book = get_ccq_book(num)
            dest_dir = os.path.join(OUTPUT_DIR, "provincial_quebec/ccq", book)
            os.makedirs(dest_dir, exist_ok=True)
            
            article_file = os.path.join(dest_dir, f"article_{num}.json")
            article_data = {
                "article": f"Article {num}",
                "code": "Code civil du Québec",
                "juridiction": "Québec (Provincial)",
                "texte": text,
                "chemin_taxonomy": f"provincial_quebec/ccq/{book}"
            }
            
            with open(article_file, "w", encoding="utf-8") as f_out:
                json.dump(article_data, f_out, ensure_ascii=False, indent=2)
                
            corpus_index.append({
                "id": f"ccq_{num}",
                "title": f"CCQ Article {num}",
                "path": article_file,
                "juridiction": "provincial_quebec"
            })
            ccq_count += 1
            
        print(f"✓ {ccq_count} articles du CCQ structurés avec succès par livre.")
    else:
        print("⚠ Fichier ccqDb.json absent de data/")
        
    # 2. Traitement du CPC
    if os.path.exists(cpc_path):
        print(f"Ingestion du Code de procédure civile du Québec depuis {cpc_path}...")
        with open(cpc_path, "r", encoding="utf-8") as f:
            cpc_data = json.load(f)
            
        cpc_count = 0
        for item in cpc_data:
            num = item["numero"]
            text = item["texte"]
            
            chapter = get_cpc_chapter(num)
            dest_dir = os.path.join(OUTPUT_DIR, "provincial_quebec/cpc", chapter)
            os.makedirs(dest_dir, exist_ok=True)
            
            article_file = os.path.join(dest_dir, f"article_{num}.json")
            article_data = {
                "article": f"Article {num}",
                "code": "Code de procédure civile du Québec",
                "juridiction": "Québec (Provincial)",
                "texte": text,
                "chemin_taxonomy": f"provincial_quebec/cpc/{chapter}"
            }
            
            with open(article_file, "w", encoding="utf-8") as f_out:
                json.dump(article_data, f_out, ensure_ascii=False, indent=2)
                
            corpus_index.append({
                "id": f"cpc_{num}",
                "title": f"CPC Article {num}",
                "path": article_file,
                "juridiction": "provincial_quebec"
            })
            cpc_count += 1
            
        print(f"✓ {cpc_count} articles du CPC structurés avec succès par chapitre.")
    else:
        print("⚠ Fichier cpcDb.json absent de data/")
        
    # Enregistrer l'index global du corpus
    if corpus_index:
        index_file = os.path.join(OUTPUT_DIR, "index.json")
        with open(index_file, "w", encoding="utf-8") as f_idx:
            json.dump(corpus_index, f_idx, ensure_ascii=False, indent=2)
        print(f"✅ Indexation globale complétée ! {len(corpus_index)} articles au total enregistrés dans {index_file}.")
        
if __name__ == "__main__":
    compile_corpus()
