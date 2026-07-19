"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadCcqDb = loadCcqDb;
exports.loadCpcDb = loadCpcDb;
exports.getArticles = getArticles;
exports.searchArticlesByKeyword = searchArticlesByKeyword;
exports.getCpcArticles = getCpcArticles;
exports.searchCpcArticlesByKeyword = searchCpcArticlesByKeyword;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
// In-memory dictionaries of articles for fast lookups
const ccqDb = new Map();
const cpcDb = new Map();
/**
 * Charge le fichier JSON extrait de l'EPUB du Code Civil.
 */
async function loadCcqDb() {
    const jsonPath = path_1.default.resolve(process.cwd(), 'data/ccqDb.json');
    if (!fs_1.default.existsSync(jsonPath)) {
        console.warn(`\n⚠️  ATTENTION : Le fichier de base de données JSON CCQ est introuvable !`);
        throw new Error('Fichier ccqDb.json manquant.');
    }
    console.log('Lecture de la base de données CCQ...');
    const jsonData = fs_1.default.readFileSync(jsonPath, 'utf8');
    const articles = JSON.parse(jsonData);
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
async function loadCpcDb() {
    const jsonPath = path_1.default.resolve(process.cwd(), 'data/cpcDb.json');
    if (!fs_1.default.existsSync(jsonPath)) {
        console.warn(`\n⚠️  ATTENTION : Le fichier de base de données JSON CPC est introuvable !`);
        throw new Error('Fichier cpcDb.json manquant.');
    }
    console.log('Lecture de la base de données CPC...');
    const jsonData = fs_1.default.readFileSync(jsonPath, 'utf8');
    const articles = JSON.parse(jsonData);
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
function getArticles(start, end) {
    const results = [];
    if (!end)
        end = start;
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
function searchArticlesByKeyword(keyword) {
    const results = [];
    const lowerKeyword = keyword.toLowerCase();
    for (const article of ccqDb.values()) {
        if (article.texte.toLowerCase().includes(lowerKeyword)) {
            results.push(article);
            if (results.length >= 50)
                break;
        }
    }
    results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
    return results;
}
/**
 * Récupère une plage d'articles du CPC
 */
function getCpcArticles(start, end) {
    const results = [];
    if (!end)
        end = start;
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
function searchCpcArticlesByKeyword(keyword) {
    const results = [];
    const lowerKeyword = keyword.toLowerCase();
    for (const article of cpcDb.values()) {
        if (article.texte.toLowerCase().includes(lowerKeyword)) {
            results.push(article);
            if (results.length >= 50)
                break;
        }
    }
    results.sort((a, b) => parseFloat(a.numero) - parseFloat(b.numero));
    return results;
}
