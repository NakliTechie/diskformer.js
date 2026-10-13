// Records llama.cpp reference replies for the chat gate: runs llama-server on a GGUF (Metal, every layer on the GPU),
// sends the 9 gate conversations greedily, and writes prompt + reply per conversation, as in examples/chat/refs/.
// --server picks another llama-server build, e.g. the webgpu-disk-tier fork's; --paged <slots> then pages the experts
// (-ot exps=WebGPU_Paged, read from the GGUF), so paged and unpaged runs of one backend can be compared
// (scripts/compare-refs.mjs).
//
//   node scripts/record-refs.mjs <model.gguf> <out.json> [--port 8799] [--ctx 4096] [--server <llama-server>] [--paged <slots>] [--ngl 0 (CPU)] [--no-stream]
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
const port = +opt('--port', 8799), ctx = +opt('--ctx', 4096), bin = opt('--server', 'llama-server'), paged = opt('--paged', null), ngl = opt('--ngl', '99');
const backend = ngl === '0' ? 'CPU' : bin === 'llama-server' ? 'Metal' : paged ? `WebGPU, experts paged (${paged} slots)` : 'WebGPU';
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

// Streamed, so the response headers arrive at once: a slow backend (a large model paged from disk) can take longer per
// reply than fetch waits for headers (5 minutes).
// --no-stream: one plain request per reply through curl (on Qwen3-Next, llama-server b9830's CPU path closed a stream
// mid-prompt, and Node's fetch dropped plain requests).
async function chat(messages, maxTokens) {
  if (rest.includes('--no-stream')) {
    // curl, not fetch: Node's fetch dropped the connection while a slow server worked on a reply.
    // Up to 3 tries: the server sometimes closes a long request with an empty reply (curl 52); greedy, so a retry is
    // the same reply.
    let out;
    for (let t = 0; t < 3; t++) {
      out = spawnSync('curl', ['-s', '--max-time', '3600', '-H', 'content-type: application/json', '-d', JSON.stringify({ messages, temperature: 0, max_tokens: maxTokens }), `${base}/v1/chat/completions`], { encoding: 'utf8', maxBuffer: 1 << 24 });
      if (out.status === 0) break;
      console.error(`curl exited ${out.status}; retrying`);
    }
    if (out.status !== 0) throw new Error(`curl exited ${out.status} three times: ${out.stderr}`);
    const r = JSON.parse(out.stdout);
    return { content: r.choices[0].message.content, finish: r.choices[0].finish_reason };
  }
  const r = await fetch(base + '/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages, temperature: 0, max_tokens: maxTokens, stream: true }) });
  if (!r.ok) throw new Error(`/v1/chat/completions: HTTP ${r.status} ${await r.text()}`);
  let content = '', finish = null, buf = '';
  const dec = new TextDecoder();
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith('data:') || line === 'data: [DONE]') continue;
      const ch = JSON.parse(line.slice(5)).choices[0];
      if (ch.delta && ch.delta.content) content += ch.delta.content;
      if (ch.finish_reason) finish = ch.finish_reason;
    }
  }
  return { content, finish };
}

const server = spawn(bin, ['-m', gguf, '-ngl', ngl, '-c', String(ctx), '--port', String(port), '--jinja', '--timeout', '3600', ...(paged ? ['-ot', 'exps=WebGPU_Paged'] : [])], {
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
    if (i > 1800) throw new Error('llama-server did not become healthy in 30 minutes');
    await new Promise((r) => setTimeout(r, 1000));
  }
  const ver = spawnSync(bin, ['--version'], { encoding: 'utf8' });  // prints on stderr
  // Release builds print "version: 9830 (c1a1c8ee9)"; a fork build "version: 0.6.0-dev (build 11473, commit 3119cf53d)".
  const vs = `${ver.stdout}${ver.stderr}`, v = vs.match(/version: (\d+) \(([0-9a-f]+)\)/), f = vs.match(/build (\d+), commit ([0-9a-f]+)/);
  const version = v ? `b${v[1]} (${v[2]})` : f ? `fork build ${f[1]} (${f[2]})` : '';
  const result = [];
  for (const c of conversations) {
    const { prompt } = await post('/apply-template', { messages: c.messages });
    const { content, finish } = await chat(c.messages, c.tokens);
    result.push({ name: c.name, messages: c.messages, prompt, content, finish, tokens: c.tokens });
    console.log(`${c.name}: ${JSON.stringify(content.slice(0, 60))}…`);
  }
  const doc = {
    about: `llama.cpp ${version || '?'}, ${backend}, -ngl ${ngl}: llama-server /v1/chat/completions, temperature 0, max_tokens 64, on ${basename(gguf)} (sha256 ${await sha256()}). Recorded ${new Date().toISOString().slice(0, 10)} by scripts/record-refs.mjs.`,
    conversations: result,
  };
  writeFileSync(out, JSON.stringify(doc, null, 1) + '\n');
  console.log(`wrote ${out}`);
} catch (e) {
  console.error(`${e.message}\nserver ${server.exitCode !== null ? `exited ${server.exitCode}` : 'still running'}, last log:\n${log.slice(-3000)}`);
  process.exitCode = 1;
} finally {
  stop();
}
