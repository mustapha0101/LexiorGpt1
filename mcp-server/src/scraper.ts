/* eslint-disable rxjs/finnish */
import axios from 'axios';
import * as cheerio from 'cheerio';
import Exa from 'exa-js';

// Initialize Exa Search client
const exaKey = process.env.EXA_API_KEY || process.env.COPILOT_EXA_KEY || '1c3736ac-d902-4089-9d5e-cbe3ef66b578';
const exa = new Exa(exaKey);

/**
 * Searches Quebec regulations using Exa restricted to the LégisQuébec regulations domain
 */
export async function searchQuebecRegulations(query: string): Promise<string> {
  console.log(`[LégisQuébec Search] Querying regulations for: "${query}"`);
  
  const result = await exa.search(query, {
    includeDomains: ['legisquebec.gouv.qc.ca/fr/document/rc'],
    contents: {
      summary: true
    },
    numResults: 8
  });

  if (!result.results || result.results.length === 0) {
    return `Aucun règlement trouvé pour la recherche "${query}" sur LégisQuébec.`;
  }

  // Filter out 404 pages
  const validResults = result.results.filter((res: any) => {
    const title = (res.title || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    return !title.includes('404') && !title.includes('non trouve') && !title.includes('erreur');
  });

  if (validResults.length === 0) {
    return `Aucun règlement valide trouvé pour "${query}" (les résultats indexés retournaient des pages d'erreur 404).`;
  }

  return validResults.map((res: any) => `
### [${res.title || 'Règlement sans titre'}](${res.url})
**Résumé :** ${res.summary || 'Aucun résumé disponible.'}
  `).join('\n\n');
}

/**
 * Searches Quebec case law (court decisions) using Exa restricted to CanLII Quebec and SOQUIJ
 */
export async function searchQuebecJurisprudence(query: string): Promise<string> {
  console.log(`[LégisQuébec Search] Querying jurisprudence for: "${query}"`);
  
  const result = await exa.search(query, {
    includeDomains: [
      'canlii.org/fr/qc',
      'citoyens.soquij.qc.ca',
      'canlii.org/en/qc'
    ],
    contents: {
      summary: true
    },
    numResults: 8
  });

  if (!result.results || result.results.length === 0) {
    return `Aucune décision de justice trouvée pour la recherche "${query}".`;
  }

  // Filter out 404 pages
  const validResults = result.results.filter((res: any) => {
    const title = (res.title || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    return !title.includes('404') && !title.includes('non trouve') && !title.includes('erreur');
  });

  if (validResults.length === 0) {
    return `Aucune décision valide trouvée pour "${query}" (les résultats indexés retournaient des pages d'erreur 404).`;
  }

  return validResults.map((res: any) => `
### [${res.title || 'Décision sans titre'}](${res.url})
**Résumé :** ${res.summary || 'Aucun résumé disponible.'}
  `).join('\n\n');
}

/**
 * Scrapes content from a specific LégisQuébec information page (modifications, dispositions non en vigueur, etc.)
 */
export async function scrapeQuebecLegalInfo(type: string): Promise<string> {
  const url = `https://www.legisquebec.gouv.qc.ca/fr/contenu/${type}`;
  console.log(`[LégisQuébec Scraper] Scraping content from: ${url}`);
  
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
      },
      timeout: 15000
    });

    const $ = cheerio.load(response.data);

    // Focus on the main content card or content div
    const contentArea = $('#contentDiv, .card-body, main, article').first();
    if (!contentArea || contentArea.length === 0) {
      return `Impossible de localiser la zone de contenu sur la page LégisQuébec : ${url}`;
    }

    // Clean up forms, scripts, sidebars, navigation lists
    contentArea.find('script, style, iframe, form, nav, .list-group, .navbar').remove();

    // Convert HTML tables to Markdown tables for readability
    contentArea.find('table').each((_, tableEl) => {
      const $table = $(tableEl);
      let markdownTable = '\n\n';

      // Parse headers
      const headers: string[] = [];
      $table.find('th').each((__, thEl) => {
        headers.push($(thEl).text().trim().replace(/\s+/g, ' '));
      });

      if (headers.length > 0) {
        markdownTable += `| ${headers.join(' | ')} |\n`;
        markdownTable += `| ${headers.map(() => '---').join(' | ')} |\n`;
      }

      // Parse rows
      $table.find('tr').each((__, trEl) => {
        const $tr = $(trEl);
        
        // Skip header row since we already parsed headers
        if ($tr.find('th').length > 0) return;

        const cells: string[] = [];
        $tr.find('td').each((___, tdEl) => {
          cells.push($(tdEl).text().trim().replace(/\s+/g, ' '));
        });

        if (cells.length > 0) {
          markdownTable += `| ${cells.join(' | ')} |\n`;
        }
      });

      $table.replaceWith(markdownTable + '\n');
    });

    // Extract text blocks
    const textBlocks: string[] = [];
    
    // Process text from paragraphs and lists
    contentArea.find('h1, h2, h3, h4, p, li, table').each((_, block) => {
      const tagName = block.tagName.toLowerCase();
      const $block = $(block);
      
      // If it's already a table replaced by markdown, push it as-is
      const text = $block.text().trim();
      if (!text) return;

      if (tagName.startsWith('h')) {
        const level = tagName.substring(1);
        textBlocks.push(`\n${'#'.repeat(Number(level))} ${text.replace(/\s+/g, ' ')}\n`);
      } else if (tagName === 'li') {
        textBlocks.push(`- ${text.replace(/\s+/g, ' ')}`);
      } else if (tagName === 'p') {
        textBlocks.push(`\n${text.replace(/\s+/g, ' ')}\n`);
      }
    });

    let markdownText = textBlocks.join('\n').trim();

    if (!markdownText) {
      // Fallback: extract cleaned raw text
      markdownText = contentArea.text()
        .replace(/\n\s*\n/g, '\n\n')
        .replace(/\s+/g, ' ')
        .trim();
    }

    if (!markdownText) {
      return `La page LégisQuébec (${url}) semble vide ou non-structurée pour le grattage direct.`;
    }

    return `# Informations LégisQuébec - ${type.toUpperCase()}\n\nSource officielle : ${url}\n\n${markdownText}`;
  } catch (err: any) {
    console.error(`[LégisQuébec Scraper] Erreur lors du scraping de ${type}:`, err.message);
    
    // Fallback: Use Exa Crawl content extraction if Axios gets blocked or fails
    try {
      console.log(`[LégisQuébec Scraper] Fallback: Tentative de récupération via Exa Crawl...`);
      const contentsResult = await exa.getContents([url], {
        text: { maxCharacters: 30000 }
      });
      if (contentsResult.results && contentsResult.results.length > 0 && contentsResult.results[0].text) {
        return `# Informations LégisQuébec - ${type.toUpperCase()}\n\nSource officielle : ${url}\n\n${contentsResult.results[0].text}`;
      }
    } catch (e: any) {
      console.error(`[LégisQuébec Scraper] Échec du fallback Exa Crawl :`, e.message);
    }
    
    throw new Error(`Impossible de récupérer les informations LégisQuébec pour ${type}. Détails : ${err.message}`);
  }
}

/**
 * Searches Canadian case law and statutes using Exa restricted to CanLII (pancanadian).
 */
export async function searchCanadianLegalDocuments(
  query: string,
  docType?: 'laws' | 'decisions' | 'all' | string,
  size: number = 8
): Promise<string> {
  console.log(`[CanLII Search] Querying Canadian documents for: "${query}" (type: ${docType || 'all'})`);

  const includeDomains = ['canlii.org'];
  
  // Refine domains depending on docType if needed
  // e.g. /fr/doc/ for legislation, /fr/csc/ for Supreme Court decisions, etc.
  // Standard canlii.org domain covers all.
  
  const result = await exa.search(query, {
    includeDomains,
    contents: {
      summary: true
    },
    numResults: size
  });

  if (!result.results || result.results.length === 0) {
    return `Aucun document juridique trouvé sur CanLII pour la recherche "${query}".`;
  }

  const validResults = result.results.filter((res: any) => {
    const title = (res.title || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    return !title.includes('404') && !title.includes('non trouve') && !title.includes('erreur');
  });

  if (validResults.length === 0) {
    return `Aucun document CanLII valide trouvé pour "${query}" (résultats en erreur 404).`;
  }

  return validResults.map((res: any) => `
### [${res.title || 'Document CanLII'}](${res.url})
**Résumé :** ${res.summary || 'Aucun résumé disponible.'}
  `).join('\n\n');
}

