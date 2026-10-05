// Assembles the Hugging Face Space (sdk: static) for examples/chat. A static Space serves its app file's folder as
// the site root, so the page goes to the root and its imports move from ../../ to ./.
//   node scripts/build-space.mjs <out-dir>
import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const here = new URL('..', import.meta.url).pathname, out = process.argv[2];
if (!out) { console.error('usage: node scripts/build-space.mjs <out-dir>'); process.exit(2); }
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
const page = (await readFile(join(here, 'examples/chat/index.html'), 'utf8'))
  .split("from '../../engines/").join("from './engines/")
  .split("from '../../src/").join("from './src/");
if (/['"]\.\.\//.test(page)) { console.error('index.html still has a ../ path the Space cannot serve'); process.exit(1); }
await writeFile(join(out, 'index.html'), page);
await cp(join(here, 'examples/chat/space-README.md'), join(out, 'README.md'));
await cp(join(here, 'examples/chat/refs'), join(out, 'refs'), { recursive: true });
for (const d of ['src', 'engines']) await cp(join(here, d), join(out, d), { recursive: true });
await cp(join(here, 'LICENSE'), join(out, 'LICENSE'));
console.log(`Space files in ${out}`);
