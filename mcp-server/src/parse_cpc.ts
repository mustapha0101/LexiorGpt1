/* eslint-disable rxjs/finnish */
import fs from 'fs';
import path from 'path';
import axios from 'axios';
import * as cheerio from 'cheerio';

const outputJson = path.resolve(__dirname, '../data/cpcDb.json');

async function parseCpc() {
  console.log('Fetching Code de procédure civile (C-25.01) from LégisQuébec...');
  const url = 'https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-25.01';
  
  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
      },
      timeout: 30000
    });

    console.log('HTML fetched successfully. Parsing...');
    const $ = cheerio.load(response.data);

    const cpcDb: { numero: string; texte: string }[] = [];

    // Find all div elements with ID starting with "se:"
    const articleDivs = $('div[id^="se:"]');
    console.log(`Found ${articleDivs.length} article elements to process.`);

    articleDivs.each((_, el) => {
      const id = $(el).attr('id');
      if (!id || id.includes('-')) return; // ignore sub-elements or section IDs with hyphens

      const articleNum = id.replace('se:', '').replace('_', '.');
      const contentBlocks: string[] = [];

      // Find all Subsection and Paragraph elements inside this article
      $(el).find('span.Subsection, div.Paragraph').each((__, blockEl) => {
        const $block = $(blockEl);
        const $clone = $block.clone();

        // Strip out metadata, links, history elements
        $clone.find('.HistoryLink, .linkOtherLang, .HistoricalNote, style, script').remove();

        let text = $clone.text().replace(/\s+/g, ' ').trim();

        // Strip the leading article number (e.g. "1." or "62.") if present at the start of the block
        const leadingNumRegex = new RegExp(`^${articleNum.replace('.', '\\.')}\\.?\\s*`);
        text = text.replace(leadingNumRegex, '');

        if (text) {
          contentBlocks.push(text);
        }
      });

      if (contentBlocks.length > 0) {
        cpcDb.push({
          numero: articleNum,
          texte: contentBlocks.join('\n')
        });
      }
    });

    console.log(`Total CPC articles parsed: ${cpcDb.length}`);

    // Create data directory if it doesn't exist
    const dataDir = path.dirname(outputJson);
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    fs.writeFileSync(outputJson, JSON.stringify(cpcDb, null, 2), 'utf8');
    console.log(`✅ Code de procédure civile database saved to: ${outputJson}`);
  } catch (err: any) {
    console.error('Error fetching or parsing CPC:', err.message);
    process.exit(1);
  }
}

parseCpc();
