// Keeps the files diskformer shares with LocalMind in step. Each file has one source:
//   src/ (the store and residency layer)  diskformer → LocalMind, byte-identical
//   engines/ (the reference engines)       LocalMind → diskformer, import paths rewritten to ../src/
//   node scripts/sync-localmind.mjs [--write] [--localmind <dir>]
// Without --write it only checks, and exits 1 when a copy differs from what its source would produce.
import { readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

const here = new URL('..', import.meta.url).pathname;
const i = process.argv.indexOf('--localmind');
const lm = i > 0 ? process.argv[i + 1] : process.env.LOCALMIND || join(homedir(), 'Code/naklios-universe/LocalMind');
const write = process.argv.includes('--write');

const STORE = [['opfs-reader.js', 'opfs-reader.js'], ['expert-stream.js', 'moe-expert-stream.js'], ['rows.js', 'rows.js'], ['gguf.js', 'gguf.js'], ['ingest.js', 'ingest.js']];
const ENGINES = ['qwen3_moe_ssd.js', 'qwen35_moe_ssd.js', 'gemma4_moe_ssd.js'];
// LocalMind's sibling paths → diskformer's src/ paths, in import and export-from statements.
const REWRITES = [['./opfs-reader.js', '../src/opfs-reader.js'], ['./moe-expert-stream.js', '../src/expert-stream.js'], ['./gguf.js', '../src/gguf.js'], ['./ingest.js', '../src/ingest.js']];
const rewrite = (text) => REWRITES.reduce((t, [a, b]) => t.split(`from '${a}'`).join(`from '${b}'`), text);

const pairs = [
  ...STORE.map(([ours, theirs]) => ({ src: join(here, 'src', ours), dst: join(lm, theirs), map: (t) => t })),
  ...ENGINES.map((f) => ({ src: join(lm, f), dst: join(here, 'engines', f), map: rewrite })),
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
