// Assembles the demo as a static site: the llama.cpp page (ggml/chat.html) at the root, the hand-written engines' page
// (examples/chat) under classic/, the run-time LoRA switch (examples/lora) under lora/. A static HF Space serves its app file's folder as the site root, and
// diskformer.naklitechie.com serves the same tree.
//   node scripts/build-space.mjs <out-dir> [--space]   --space adds the Space's README (HF metadata)
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const here = new URL('..', import.meta.url).pathname, out = process.argv[2], space = process.argv.includes('--space');
if (!out) { console.error('usage: node scripts/build-space.mjs <out-dir> [--space]'); process.exit(2); }
await rm(out, { recursive: true, force: true });
await mkdir(join(out, 'classic'), { recursive: true });
const noParent = (name, text) => { if (/['"]\.\.\//.test(text)) { console.error(`${name} still has a ../ path the site cannot serve`); process.exit(1); } return text; };

// The root: the llama.cpp page (ggml/chat.html), its worker and the wasm build under ggml/, the store layer under src/.
const page = (await readFile(join(here, 'ggml/chat.html'), 'utf8')).split("new URL('./worker.mjs', import.meta.url)").join("new URL('./ggml/worker.mjs', import.meta.url)");
await writeFile(join(out, 'index.html'), noParent('ggml/chat.html', page));
await mkdir(join(out, 'ggml'), { recursive: true });
await cp(join(here, 'ggml/worker.mjs'), join(out, 'ggml/worker.mjs'));   // imports ./dist/ and ../src/, both in the tree
await cp(join(here, 'ggml/dist'), join(out, 'ggml/dist'), { recursive: true });
await cp(join(here, 'src'), join(out, 'src'), { recursive: true });

// classic/: the hand-written engines' page (examples/chat), its imports moved from ../../ to ./.
const classic = (await readFile(join(here, 'examples/chat/index.html'), 'utf8'))
  .split("from '../../engines/").join("from './engines/")
  .split("from '../../src/").join("from './src/");
await writeFile(join(out, 'classic/index.html'), noParent('examples/chat/index.html', classic));
await cp(join(here, 'examples/chat/refs'), join(out, 'classic/refs'), { recursive: true });
for (const d of ['src', 'engines']) await cp(join(here, d), join(out, 'classic', d), { recursive: true });

// lora/: the run-time LoRA switch (examples/lora), on the engines copied into classic/.
const lora = (await readFile(join(here, 'examples/lora/index.html'), 'utf8')).split("from '../../engines/").join("from '../classic/engines/");
await mkdir(join(out, 'lora'), { recursive: true });
// lora/ sits one level below the site root, so ../classic/ stays in the tree and only ../../ leaves it
if (/['"]\.\.\/\.\.\//.test(lora)) { console.error('examples/lora/index.html still has a ../../ path the site cannot serve'); process.exit(1); }
await writeFile(join(out, 'lora/index.html'), lora);

if (space) await cp(join(here, 'examples/chat/space-README.md'), join(out, 'README.md'));
await cp(join(here, 'LICENSE'), join(out, 'LICENSE'));
await cp(join(here, 'marketing/social.png'), join(out, 'social.png'));  // the og:image / twitter:image card
console.log(`${space ? 'Space' : 'site'} files in ${out}`);
