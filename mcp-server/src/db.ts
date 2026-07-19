import fs from 'fs';
import path from 'path';

export interface Article {
  numero: string;
  texte: string;
}

// In-memory dictionaries of articles for fast lookups
const ccqDb: Map<number, Article> = new Map();
const cpcDb: Map<number, Article> = new Map();

/**
 * Charge le fichier JSON extrait de l'EPUB du Code Civil.
 */
export async function loadCcqDb() {
  const jsonPath = path.resolve(process.cwd(), 'data/ccqDb.json');
  
  if (!fs.existsSync(jsonPath)) {
    console.warn(`\n⚠️  ATTENTION : Le fichier de base de données JSON CCQ est introuvable !`);
    throw new Error('Fichier ccqDb.json manquant.');
  }

  console.log('Lecture de la base de données CCQ...');
  const jsonData = fs.readFileSync(jsonPath, 'utf8');
  const articles: Article[] = JSON.parse(jsonData);

  ccqDb.clear();
  for (const article of articles) {
    const numParse = parseFloat(article.numero);
    if (!isNaN(numParse) && article.texte) {
      ccqDb.set(numParse, { numero: article.numero, texte: article.texte });
    }
  }

  console.log(`✅ Extraction CCQ terminée : ${ccqDb.size} articles indexés.`);
}

/**
 * Charge le fichier JSON extrait de LégisQuébec pour le Code de procédure civile.
 */
export async function loadCpcDb() {
  const jsonPath = path.resolve(process.cwd(), 'data/cpcDb.json');
  
  if (!fs.existsSync(jsonPath)) {
    console.warn(`\n⚠️  ATTENTION : Le fichier de base de données JSON CPC est introuvable !`);
    throw new Error('Fichier cpcDb.json manquant.');
  }

  console.log('Lecture de la base de données CPC...');
  const jsonData = fs.readFileSync(jsonPath, 'utf8');
  const articles: Article[] = JSON.parse(jsonData);

  cpcDb.clear();
  for (const article of articles) {
    const numParse = parseFloat(article.numero);
    if (!isNaN(numParse) && article.texte) {
      cpcDb.set(numParse, { numero: article.numero, texte: article.texte });
    }
  }

  console.log(`✅ Extraction CPC terminée : ${cpcDb.size} articles indexés.`);
}

/**
 * Récupère une plage d'articles du CCQ
 */
export function getArticles(start: number, end?: number): Article[] {
  const results: Article[] = [];
  if (!end) end = start;
  
  for (const [num, article] of ccqDb.entries()) {
    if (num >= start && num <= end) {
      results.push(article);
    }
  }

  results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
  return results;
}

/**
 * Recherche des articles du CCQ par mot-clé
 */
export function searchArticlesByKeyword(keyword: string): Article[] {
  const results: Article[] = [];
  const lowerKeyword = keyword.toLowerCase();
  
  for (const article of ccqDb.values()) {
    if (article.texte.toLowerCase().includes(lowerKeyword)) {
      results.push(article);
      if (results.length >= 50) break;
    }
  }

  results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
  return results;
}

/**
 * Récupère une plage d'articles du CPC
 */
export function getCpcArticles(start: number, end?: number): Article[] {
  const results: Article[] = [];
  if (!end) end = start;
  
  for (const [num, article] of cpcDb.entries()) {
    if (num >= start && num <= end) {
      results.push(article);
    }
  }

  results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
  return results;
}

/**
 * Recherche des articles du CPC par mot-clé
 */
export function searchCpcArticlesByKeyword(keyword: string): Article[] {
  const results: Article[] = [];
  const lowerKeyword = keyword.toLowerCase();
  
  for (const article of cpcDb.values()) {
    if (article.texte.toLowerCase().includes(lowerKeyword)) {
      results.push(article);
      if (results.length >= 50) break;
    }
  }

  results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
  return results;
}
