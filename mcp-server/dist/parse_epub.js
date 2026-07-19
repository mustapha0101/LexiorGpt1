"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/* eslint-disable rxjs/finnish */
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const cheerio = __importStar(require("cheerio"));
const extractedDir = path_1.default.resolve(__dirname, '../data/epub_extracted/OPF');
const outputJson = path_1.default.resolve(__dirname, '../data/ccqDb.json');
const ccqDb = [];
// Lire tous les fichiers pageX.xhtml
const files = fs_1.default
    .readdirSync(extractedDir)
    .filter(f => f.startsWith('page') && f.endsWith('.xhtml'))
    .sort((a, b) => {
    // page1.xhtml, page2.xhtml -> tri numérique
    const numA = parseInt(a.replace('page', '').replace('.xhtml', ''));
    const numB = parseInt(b.replace('page', '').replace('.xhtml', ''));
    return numA - numB;
});
console.log(`Trouvé ${files.length} fichiers XHTML à analyser...`);
for (const file of files) {
    const filePath = path_1.default.join(extractedDir, file);
    const html = fs_1.default.readFileSync(filePath, 'utf8');
    // Utiliser cheerio
    const $ = cheerio.load(html, { xmlMode: true });
    // Tous les div avec un id de type "se:NUMERO" ou "se:NUMERO_DECIMAL"
    const articleDivs = $('div[id^="se:"]');
    articleDivs.each((_, el) => {
        const id = $(el).attr('id');
        if (!id || id.includes('-'))
            return; // ignore sub-sections
        // Le numéro est après "se:"
        let numeroStr = id.replace('se:', '').replace('_', '.'); // ex: 30_1 -> 30.1
        // Pour chaque sous-section (ss:1, ss:2, etc), extraire le texte
        // Mais on peut simplement extraire tous les spans avec texte
        // Le texte des alinéas est généralement dans des <span> qui ne contiennent pas que le numéro.
        // Un moyen plus simple: trouver tous les text nodes dans le div
        // Mais on veut exclure le numéro de l'article lui-même (ex: "1.")
        // Chercher les divs d'alinéas : div[id^="se:X-ss:"]
        const alineas = [];
        $(el)
            .find('div[id*="-ss:"]')
            .each((_, ss) => {
            // Extraire le texte de cet alinéa
            // On retire le texte du numéro d'article (ex: "1.")
            $(ss).find('span[style*="font-weight: bold"]').remove(); // Retire le span du numéro
            let text = $(ss).text().trim();
            // Parfois il reste des espaces en trop ou des puces, on nettoie
            text = text.replace(/\s+/g, ' ').trim();
            if (text) {
                alineas.push(text);
            }
        });
        if (alineas.length > 0) {
            ccqDb.push({
                numero: numeroStr,
                texte: alineas.join('\n'),
            });
        }
    });
    console.log(`Fichier ${file} analysé.`);
}
console.log(`Total d'articles extraits : ${ccqDb.length}`);
// Sauvegarder en JSON
fs_1.default.writeFileSync(outputJson, JSON.stringify(ccqDb, null, 2), 'utf8');
console.log(`✅ Base de données sauvegardée dans : ${outputJson}`);
