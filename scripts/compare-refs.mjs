// Compares the replies of two recordings of the chat gate (scripts/record-refs.mjs), conversation by conversation:
// identical, or the character where they part. Exit 1 when any reply differs.
//   node scripts/compare-refs.mjs <a.json> <b.json>
import { readFileSync } from 'node:fs';

const [a, b] = process.argv.slice(2).map((f) => JSON.parse(readFileSync(f, 'utf8')));
let same = 0;
for (const x of a.conversations) {
  const y = b.conversations.find((c) => c.name === x.name);
  if (!y) { console.log(`${x.name}: missing in the second file`); continue; }
  let i = 0;
  while (i < x.content.length && x.content[i] === y.content[i]) i++;
  if (i === x.content.length && i === y.content.length) { same++; console.log(`${x.name}: identical`); continue; }
  console.log(`${x.name}: differs at char ${i}: ${JSON.stringify(x.content.slice(Math.max(0, i - 20), i + 20))} vs ${JSON.stringify(y.content.slice(Math.max(0, i - 20), i + 20))}`);
}
console.log(`${same} of ${a.conversations.length} identical`);
process.exit(same === a.conversations.length ? 0 : 1);
