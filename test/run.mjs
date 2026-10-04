// Runs every Node test in this folder; exits non-zero on the first failure.   npm test
import { readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const dir = new URL('.', import.meta.url).pathname;
for (const f of readdirSync(dir).filter((n) => n.endsWith('.test.mjs')).sort()) {
  process.stdout.write(execFileSync(process.execPath, [dir + f], { encoding: 'utf8' }));
}
