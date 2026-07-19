/* eslint-disable rxjs/finnish */
import fs from 'fs';
import path from 'path';
import * as cheerio from 'cheerio';

const extractedDir = path.resolve(__dirname, '../data/epub_extracted/OPF');
const outputJson = path.resolve(__dirname, '../data/ccqDb.json');

const ccqDb: { numero: string; texte: string }[] = [];

// Lire tous les fichiers pageX.xhtml
const files = fs
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
  const filePath = path.join(extractedDir, file);
  const html = fs.readFileSync(filePath, 'utf8');

  // Utiliser cheerio
  const $ = cheerio.load(html, { xmlMode: true });

  // Tous les div avec un id de type "se:NUMERO" ou "se:NUMERO_DECIMAL"
  const articleDivs = $('div[id^="se:"]');

  articleDivs.each((_: any, el: any) => {
    const id = $(el).attr('id');
    if (!id || id.includes('-')) return; // ignore sub-sections

    // Le numéro est après "se:"
    let numeroStr = id.replace('se:', '').replace('_', '.'); // ex: 30_1 -> 30.1

    // Pour chaque sous-section (ss:1, ss:2, etc), extraire le texte
    // Mais on peut simplement extraire tous les spans avec texte
    // Le texte des alinéas est généralement dans des <span> qui ne contiennent pas que le numéro.

    // Un moyen plus simple: trouver tous les text nodes dans le div
    // Mais on veut exclure le numéro de l'article lui-même (ex: "1.")

    // Chercher les divs d'alinéas : div[id^="se:X-ss:"]
    const alineas: string[] = [];
    $(el)
      .find('div[id*="-ss:"]')
      .each((_: any, ss: any) => {
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
fs.writeFileSync(outputJson, JSON.stringify(ccqDb, null, 2), 'utf8');
console.log(`✅ Base de données sauvegardée dans : ${outputJson}`);
