// Records llama.cpp reference replies for the chat gate: runs llama-server on a GGUF (Metal, every layer on the GPU),
// sends the 9 gate conversations greedily, and writes prompt + reply per conversation, as in examples/chat/refs/.
//
//   node scripts/record-refs.mjs <model.gguf> <out.json> [--port 8799] [--ctx 4096]
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

const [gguf, out, ...rest] = process.argv.slice(2);
if (!gguf || !out) {
  console.error('usage: node scripts/record-refs.mjs <model.gguf> <out.json> [--port 8799] [--ctx 4096]');
  process.exit(2);
}
const opt = (name, dflt) => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : dflt);
const port = +opt('--port', 8799), ctx = +opt('--ctx', 4096);
const base = `http://127.0.0.1:${port}`;
const conversations = JSON.parse(readFileSync(new URL('../examples/chat/refs/gemma.json', import.meta.url))).conversations;

const sha256 = () => new Promise((resolve, reject) => {
  const h = createHash('sha256');
  createReadStream(gguf).on('data', (d) => h.update(d)).on('end', () => resolve(h.digest('hex'))).on('error', reject);
});
const post = async (path, body) => {
  const r = await fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
  return r.json();
};

const server = spawn('llama-server', ['-m', gguf, '-ngl', '99', '-c', String(ctx), '--port', String(port), '--jinja'], { stdio: ['ignore', 'ignore', 'pipe'] });
let log = '';
server.stderr.on('data', (d) => { log += d; });
const stop = () => server.kill('SIGTERM');
process.on('exit', stop);

try {
  for (let i = 0; ; i++) {
    if (server.exitCode !== null) throw new Error(`llama-server exited:\n${log.slice(-2000)}`);
    const ok = await fetch(`${base}/health`).then((r) => r.ok).catch(() => false);
    if (ok) break;
    if (i > 600) throw new Error('llama-server did not become healthy in 10 minutes');
    await new Promise((r) => setTimeout(r, 1000));
  }
  const version = (log.match(/build: (\d+) \(([0-9a-f]+)\)/) || []).slice(1).join(' ');
  const result = [];
  for (const c of conversations) {
    const { prompt } = await post('/apply-template', { messages: c.messages });
    const r = await post('/v1/chat/completions', { messages: c.messages, temperature: 0, max_tokens: c.tokens });
    const content = r.choices[0].message.content;
    result.push({ name: c.name, messages: c.messages, prompt, content, finish: r.choices[0].finish_reason, tokens: c.tokens });
    console.log(`${c.name}: ${JSON.stringify(content.slice(0, 60))}…`);
  }
  const doc = {
    about: `llama.cpp ${version || '?'}, Metal, -ngl 99: llama-server /v1/chat/completions, temperature 0, max_tokens 64, on ${basename(gguf)} (sha256 ${await sha256()}). Recorded ${new Date().toISOString().slice(0, 10)} by scripts/record-refs.mjs.`,
    conversations: result,
  };
  writeFileSync(out, JSON.stringify(doc, null, 1) + '\n');
  console.log(`wrote ${out}`);
} finally {
  stop();
}
