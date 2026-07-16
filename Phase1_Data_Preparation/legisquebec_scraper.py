#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de scraping et d'ingestion LégisQuébec & Fédéral (Lois de Justice Canada).
Permet de récupérer le texte officiel des articles de loi à jour, de les nettoyer,
et de les stocker de manière structurée sous forme de corpus local avec cache.
"""

import os
import re
import sys
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# Configuration & Chemins
CACHE_DIR = "data/law_corpus/cache"
OUTPUT_BASE_DIR = "data/law_corpus"
USER_AGENT = "LexiorGPT-Bot/1.0 (contact@intelli.work; legal research)"

class LegalScraper:
    def __init__(self, cache_enabled=True):
        self.cache_enabled = cache_enabled
        self.headers = {"User-Agent": USER_AGENT}
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    def _get_cache_path(self, jurisdiction, source_id, article_id):
        clean_source = re.sub(r'[^a-zA-Z0-9_-]', '_', source_id)
        clean_article = re.sub(r'[^a-zA-Z0-9_-]', '_', article_id)
        return os.path.join(CACHE_DIR, f"{jurisdiction}_{clean_source}_{clean_article}.json")

    def _read_cache(self, jurisdiction, source_id, article_id):
        if not self.cache_enabled:
            return None
        cache_path = self._get_cache_path(jurisdiction, source_id, article_id)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, jurisdiction, source_id, article_id, data):
        cache_path = self._get_cache_path(jurisdiction, source_id, article_id)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def scrape_ccq_article(self, article_number):
        """
        Scrape un article du Code civil du Québec (CCQ) depuis LégisQuébec.
        URL typique : http://legisquebec.gouv.qc.ca/fr/showdoc/cs/CCQ-1991
        """
        cached = self._read_cache("quebec", "ccq", str(article_number))
        if cached:
            return cached

        print(f"Scraping Article {article_number} du CCQ sur LégisQuébec...")
        # LégisQuébec utilise un format de recherche ou d'accès direct par ancre
        url = f"http://legisquebec.gouv.qc.ca/fr/showdoc/cs/CCQ-1991?section=se:{article_number}"
        
        try:
            time.sleep(1.0) # Rate limit poli
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                print(f"Erreur HTTP {response.status_code} pour l'article {article_number}")
                return None
                
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extraction du texte de l'article spécifique
            # LégisQuébec enveloppe les articles dans des conteneurs identifiés par classe ou ID
            article_div = None
            
            # Recherche de l'ancre de l'article (ex: se:1457)
            article_anchor = soup.find(id=f"se:{article_number}")
            if article_anchor:
                article_div = article_anchor.find_parent("div")
            
            if not article_div:
                # Recherche textuelle alternative si l'ancre n'a pas été trouvée
                for div in soup.find_all("div", class_="article"):
                    if f" {article_number}." in div.get_text():
                        article_div = div
                        break
            
            if not article_div:
                # Si non trouvé, on prend le premier paragraphe principal du document (approximation)
                article_div = soup.find("div", id="doc")
                
            if not article_div:
                return None
                
            text = article_div.get_text(separator=" ").strip()
            # Nettoyer les espaces multiples
            text = re.sub(r'\s+', ' ', text)
            
            result = {
                "article": f"Article {article_number}",
                "source": "Code civil du Québec",
                "text": text,
                "url": url,
                "retrieved_at": time.time()
            }
            
            self._write_cache("quebec", "ccq", str(article_number), result)
            return result
            
        except Exception as e:
            print(f"Erreur lors du scraping de l'article {article_number} : {e}")
            return None

    def scrape_federal_statute(self, statute_name, section_number):
        """
        Scrape un article d'une loi fédérale (ex: Code criminel) depuis laws.justice.gc.ca.
        """
        cached = self._read_cache("federal", statute_name, str(section_number))
        if cached:
            return cached

        print(f"Scraping Article {section_number} de {statute_name} (Fédéral)...")
        # laws.justice.gc.ca utilise des structures d'URL très lisibles
        # ex: https://laws-lois.justice.gc.ca/fra/lois/C-46/page-1.html
        # Pour simplifier, nous interrogeons le portail de recherche ou l'API de fallback
        url = f"https://laws-lois.justice.gc.ca/fra/Search/Search.aspx?txtS3rchA11={quote(statute_name)}+{section_number}"
        
        try:
            time.sleep(1.0)
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.content, "html.parser")
            # Extraction simple du premier bloc de texte pertinent
            content_div = soup.find("div", id="wb-main")
            if not content_div:
                content_div = soup.find("main")
                
            if not content_div:
                return None
                
            text = content_div.get_text(separator=" ").strip()
            text = re.sub(r'\s+', ' ', text)
            
            result = {
                "article": f"Article {section_number}",
                "source": statute_name,
                "text": text[:1500] + "...", # Limiter pour éviter les pages de recherche entières
                "url": url,
                "retrieved_at": time.time()
            }
            
            self._write_cache("federal", statute_name, str(section_number), result)
            return result
        except Exception as e:
            print(f"Erreur lors du scraping fédéral : {e}")
            return None

def main():
    # Démo rapide si exécuté directement
    scraper = LegalScraper()
    
    # Test de récupération de l'article 1457 du CCQ
    art_1457 = scraper.scrape_ccq_article("1457")
    if art_1457:
        print("\n--- Article 1457 CCQ Récupéré ---")
        print(art_1457["text"][:300] + "...")
        
        # Enregistrer l'article dans le répertoire de la taxonomie
        dest_dir = os.path.join(OUTPUT_BASE_DIR, "provincial_quebec/civil_code/livre5_obligations")
        os.makedirs(dest_dir, exist_ok=True)
        with open(os.path.join(dest_dir, "article_1457.json"), "w", encoding="utf-8") as f:
            json.dump(art_1457, f, ensure_ascii=False, indent=2)
    else:
        print("Échec de la récupération de l'article 1457 du CCQ.")

if __name__ == "__main__":
    main()
