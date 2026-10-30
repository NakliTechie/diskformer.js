// Keeps diskformer's copy of kohra's block-diffusion loop (kohra.js denoise.js) byte-identical to its source. kohra
// owns the loop (one copy for kohra's ONNX path and diskformer's clover_moe engine); diskformer never edits it.
//   node scripts/sync-kohra.mjs [--write] [--kohra <dir>]
// Without --write it checks: the copy must match PINNED, and the kohra checkout's file when it is present.
// After a kohra release, run with --write and update PINNED to the new file's sha256 and tag.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join } from 'node:path';

const PINNED = { sha256: '4e34e9db029de389d797ce946cec2bc137f9062c236117af57c3912acf316a29', from: 'kohra v0.4.0' };
const here = new URL('..', import.meta.url).pathname;
const i = process.argv.indexOf('--kohra');
const kohra = i > 0 ? process.argv[i + 1] : process.env.KOHRA || join(homedir(), 'Code/kohra');
const copy = join(here, 'engines/vendor/kohra-denoise.js'), src = join(kohra, 'denoise.js');
const sha = (b) => createHash('sha256').update(b).digest('hex');
const ours = await readFile(copy).catch(() => null);
const theirs = await readFile(src).catch(() => null);
if (process.argv.includes('--write')) {
  if (!theirs) { console.error(`no ${src}`); process.exit(1); }
  await writeFile(copy, theirs);
  console.log(`wrote ${copy} (sha256 ${sha(theirs)}); PINNED is ${PINNED.sha256}`);
  process.exit(sha(theirs) === PINNED.sha256 ? 0 : 1);
}
let bad = 0;
if (!ours) { console.error(`missing ${copy}`); bad++; }
else if (sha(ours) !== PINNED.sha256) { console.error(`${copy}: sha256 ${sha(ours)}, pinned ${PINNED.sha256} (${PINNED.from})`); bad++; }
if (ours && theirs && !ours.equals(theirs)) { console.error(`${copy} differs from ${src}`); bad++; }
console.log(bad ? 'kohra sync: FAIL' : `kohra sync: ok (${PINNED.from}${theirs ? ', matches the checkout' : ''})`);
process.exit(bad ? 1 : 0);
