// Compares two token-level recordings of the same prompts (LocalMind's scripts/llama-ref.mjs: greedy token ids plus the
// top-5 log-probabilities at each step), e.g. llama.cpp Metal against the WebGPU fork. Per prompt: where the greedy
// tokens first part, how close the first recording's top two were there (its margin), and the largest top-1
// log-probability difference before that point. A divergence at a small margin is a near-tie between backends'
// rounding; one at a large margin is a real numeric error. Exit 1 when any divergence has a margin above --tie.
//   node scripts/compare-tokens.mjs <a.json> <b.json> [--tie 0.5]
import { readFileSync } from 'node:fs';

const [fa, fb, ...rest] = process.argv.slice(2);
const tie = Number(rest.includes('--tie') ? rest[rest.indexOf('--tie') + 1] : 0.5);
const [a, b] = [fa, fb].map((f) => JSON.parse(readFileSync(f, 'utf8')));
let same = 0, bad = 0, maxd = 0;
for (const x of a.results) {
  const y = b.results.find((r) => r.name === x.name);
  // The recordings run with EOS ignored; where either side's own top-1 is not its sampled token (the end of the reply,
  // banned), the reply has ended and the comparison stops. Builds differ in whether the banned token is listed.
  let n = Math.min(x.gen.length, y.gen.length), div = -1;
  for (let i = 0; i < n; i++) if (x.top[i][0][0] !== x.gen[i] || y.top[i][0][0] !== y.gen[i]) { n = i; break; }
  for (let i = 0; i < n; i++) if (x.gen[i] !== y.gen[i]) { div = i; break; }
  let d = 0;
  for (let i = 0; i < (div < 0 ? n : div); i++) d = Math.max(d, Math.abs(x.top[i][0][1] - y.top[i][0][1]));
  maxd = Math.max(maxd, d);
  if (div < 0) { same++; console.log(`${x.name}: identical, ${n} tokens, max top-1 |dlp| ${d.toFixed(4)}`); continue; }
  const margin = x.top[div][0][1] - x.top[div][1][1], other = y.top[div][0][1] - y.top[div][1][1];
  if (margin > tie) bad++;
  console.log(`${x.name}: parts at token ${div}, margin ${margin.toFixed(4)} (second: ${other.toFixed(4)})${margin > tie ? ' ABOVE TIE' : ''}; max top-1 |dlp| before ${d.toFixed(4)}`);
}
console.log(`${same} of ${a.results.length} identical; ${bad} divergences above a ${tie} margin; max top-1 |dlp| ${maxd.toFixed(4)}`);
process.exit(bad ? 1 : 0);
