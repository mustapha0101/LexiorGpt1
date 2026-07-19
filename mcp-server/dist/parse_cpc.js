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
const axios_1 = __importDefault(require("axios"));
const cheerio = __importStar(require("cheerio"));
const outputJson = path_1.default.resolve(__dirname, '../data/cpcDb.json');
async function parseCpc() {
    console.log('Fetching Code de procédure civile (C-25.01) from LégisQuébec...');
    const url = 'https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-25.01';
    try {
        const response = await axios_1.default.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            },
            timeout: 30000
        });
        console.log('HTML fetched successfully. Parsing...');
        const $ = cheerio.load(response.data);
        const cpcDb = [];
        // Find all div elements with ID starting with "se:"
        const articleDivs = $('div[id^="se:"]');
        console.log(`Found ${articleDivs.length} article elements to process.`);
        articleDivs.each((_, el) => {
            const id = $(el).attr('id');
            if (!id || id.includes('-'))
                return; // ignore sub-elements or section IDs with hyphens
            const articleNum = id.replace('se:', '').replace('_', '.');
            const contentBlocks = [];
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
        const dataDir = path_1.default.dirname(outputJson);
        if (!fs_1.default.existsSync(dataDir)) {
            fs_1.default.mkdirSync(dataDir, { recursive: true });
        }
        fs_1.default.writeFileSync(outputJson, JSON.stringify(cpcDb, null, 2), 'utf8');
        console.log(`✅ Code de procédure civile database saved to: ${outputJson}`);
    }
    catch (err) {
        console.error('Error fetching or parsing CPC:', err.message);
        process.exit(1);
    }
}
parseCpc();
