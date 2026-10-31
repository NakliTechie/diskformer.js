// Keeps LocalMind's copies of diskformer's files in step. diskformer is the source of every file (Chirag 2026-10-22:
// diskformer holds every custom engine; LocalMind loads models through anyformer):
//   src/ (the store and residency layer)  diskformer → LocalMind src/, byte-identical
//   engines/ (the engines)                diskformer → LocalMind src/, import paths rewritten from ../src/ to ./
// LocalMind keeps these modules in its src/ (it rolls them into its one index.html with scripts/roll-in.mjs) until it
// loads every model through anyformer.
//   node scripts/sync-localmind.mjs [--write] [--localmind <dir>]
// Without --write it only checks, and exits 1 when a copy differs from what its source would produce.
import { readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

const here = new URL('..', import.meta.url).pathname;
const i = process.argv.indexOf('--localmind');
const lm = i > 0 ? process.argv[i + 1] : process.env.LOCALMIND || join(homedir(), 'Code/naklios-universe/LocalMind');
const write = process.argv.includes('--write');
const lmSrc = join(lm, 'src');

const STORE = [['opfs-reader.js', 'opfs-reader.js'], ['expert-stream.js', 'moe-expert-stream.js'], ['rows.js', 'rows.js'], ['gguf.js', 'gguf.js'], ['ingest.js', 'ingest.js'], ['errors.js', 'errors.js']];
// MoE paging, dense layer streaming, the Bonsai 2 drafters, the FlashMap drafter, and the shared kernels they import.
const ENGINES = ['qwen3_moe_ssd.js', 'qwen35_moe_ssd.js', 'gemma4_moe_ssd.js', 'minicpmv_vision.js', 'lfm2_moe_ssd.js',
  'hy_v3_moe_ssd.js', 'flint_moe_ssd.js', 'iq_quants.js', 'flashmap_drafter.js', 'qwen35_dense.js', 'ple-opfs.js',
  'ternary_bonsai_2_dflash.js', 'ternary_bonsai_2_mtp.js', 'ling_moe_ssd.js', 'subgroup-matrix-guard.js',
  // from webml-community Hugging Face Spaces whose code Chirag wrote and released under MIT (2026-10-23)
  'lfm2_5.js', 'gemma-4-e2b.js', 'ternary_bonsai_2_27b.js'];
// diskformer's src/ paths → LocalMind's sibling paths, in import and export-from statements.
const STORE_PATHS = Object.fromEntries(STORE.map(([ours, theirs]) => [`../src/${ours}`, `./${theirs}`]));
// Module specifiers only: `from '…'`, `import '…'`, `import('…')`, either quote. A relative specifier that is
// neither a store file nor another engine would not resolve in LocalMind's src/, so it stops the sync.
const SPEC = /(\bfrom\s*|\bimport\s*\(?\s*)(['"])(\.\.?\/[^'"]+)\2/g;
function toLocalMind(text, file) {
  return text.replace(SPEC, (all, lead, q, spec) => {
    if (STORE_PATHS[spec]) return `${lead}${q}${STORE_PATHS[spec]}${q}`;
    if (spec.startsWith('./') && ENGINES.includes(spec.slice(2))) return all;
    throw new Error(`${file}: imports ${spec}, which LocalMind has no copy of; add it to the sync or remove the import`);
  });
}

const pairs = [
  ...STORE.map(([ours, theirs]) => ({ src: join(here, 'src', ours), dst: join(lmSrc, theirs), map: (t) => t })),
  ...ENGINES.map((f) => ({ src: join(here, 'engines', f), dst: join(lmSrc, f), map: (t) => toLocalMind(t, f) })),
];
let stale = 0;
for (const { src, dst, map } of pairs) {
  const want = map(await readFile(src, 'utf8'));
  const have = await readFile(dst, 'utf8').catch(() => null);
  if (have === want) continue;
  stale++;
  if (write) { await writeFile(dst, want); console.log(`wrote ${dst}`); } else console.log(`differs: ${dst} (source ${src})`);
}
if (!write && stale) { console.log(`${stale} of ${pairs.length} copies differ; run with --write to update them from their sources`); process.exit(1); }
console.log(`${pairs.length} copies match their sources${write && stale ? ` (${stale} updated)` : ''}`);
