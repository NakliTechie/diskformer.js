// Records llama.cpp reference replies for the chat gate: runs llama-server on a GGUF (Metal, every layer on the GPU),
// sends the 9 gate conversations greedily, and writes prompt + reply per conversation, as in examples/chat/refs/.
// --server picks another llama-server build, e.g. the webgpu-disk-tier fork's; --paged <slots> then pages the experts
// (-ot exps=WebGPU_Paged, read from the GGUF), so paged and unpaged runs of one backend can be compared
// (scripts/compare-refs.mjs).
//
//   node scripts/record-refs.mjs <model.gguf> <out.json> [--port 8799] [--ctx 4096] [--server <llama-server>] [--paged <slots>]
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

const [gguf, out, ...rest] = process.argv.slice(2);
if (!gguf || !out) {
  console.error('usage: node scripts/record-refs.mjs <model.gguf> <out.json> [--port 8799] [--ctx 4096]');
  process.exit(2);
}
const opt = (name, dflt) => (rest.includes(name) ? rest[rest.indexOf(name) + 1] : dflt);
const port = +opt('--port', 8799), ctx = +opt('--ctx', 4096), bin = opt('--server', 'llama-server'), paged = opt('--paged', null);
const backend = bin === 'llama-server' ? 'Metal' : paged ? `WebGPU, experts paged (${paged} slots)` : 'WebGPU';
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

const server = spawn(bin, ['-m', gguf, '-ngl', '99', '-c', String(ctx), '--port', String(port), '--jinja', ...(paged ? ['-ot', 'exps=WebGPU_Paged'] : [])], {
  stdio: ['ignore', 'ignore', 'pipe'],
  env: paged ? { ...process.env, GGML_WEBGPU_PAGED_SLOTS: paged, GGML_WEBGPU_PAGED_FILE: gguf } : process.env,
});
let log = '', failed = null;
server.stderr.on('data', (d) => { log += d; if (!failed && /ggml_abort|GGML_ASSERT|timed out/.test(String(d))) failed = String(d).trim().split('\n')[0]; });
server.on('error', (e) => { failed = e.message; });
const stop = () => server.kill('SIGTERM');
process.on('exit', stop);

try {
  for (let i = 0; ; i++) {
    if (server.exitCode !== null || failed) throw new Error(`llama-server failed (${failed || `exit ${server.exitCode}`}):\n${log.slice(-2000)}`);
    const ok = await fetch(`${base}/health`).then((r) => r.ok).catch(() => false);
    if (ok) break;
    if (i > 600) throw new Error('llama-server did not become healthy in 10 minutes');
    await new Promise((r) => setTimeout(r, 1000));
  }
  const ver = spawnSync(bin, ['--version'], { encoding: 'utf8' });  // prints on stderr
  // Release builds print "version: 9830 (c1a1c8ee9)"; a fork build "version: 0.6.0-dev (build 11473, commit 3119cf53d)".
  const vs = `${ver.stdout}${ver.stderr}`, v = vs.match(/version: (\d+) \(([0-9a-f]+)\)/), f = vs.match(/build (\d+), commit ([0-9a-f]+)/);
  const version = v ? `b${v[1]} (${v[2]})` : f ? `fork build ${f[1]} (${f[2]})` : '';
  const result = [];
  for (const c of conversations) {
    const { prompt } = await post('/apply-template', { messages: c.messages });
    const r = await post('/v1/chat/completions', { messages: c.messages, temperature: 0, max_tokens: c.tokens });
    const content = r.choices[0].message.content;
    result.push({ name: c.name, messages: c.messages, prompt, content, finish: r.choices[0].finish_reason, tokens: c.tokens });
    console.log(`${c.name}: ${JSON.stringify(content.slice(0, 60))}…`);
  }
  const doc = {
    about: `llama.cpp ${version || '?'}, ${backend}, -ngl 99: llama-server /v1/chat/completions, temperature 0, max_tokens 64, on ${basename(gguf)} (sha256 ${await sha256()}). Recorded ${new Date().toISOString().slice(0, 10)} by scripts/record-refs.mjs.`,
    conversations: result,
  };
  writeFileSync(out, JSON.stringify(doc, null, 1) + '\n');
  console.log(`wrote ${out}`);
} finally {
  stop();
}
