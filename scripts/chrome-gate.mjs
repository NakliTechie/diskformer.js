// The Chrome chat gate on the llama.cpp path, unattended: headless Chrome opens ggml/gate.html, which copies a local
// GGUF into the profile's OPFS over HTTP Range (served here from --models), replays llama.cpp's replies
// (examples/chat/refs/*.json) with paged experts, and reports how many match. Each run uses a new Chrome profile that
// is deleted when Chrome exits (Chrome holds the OPFS file until then, so removing it from a page fails), unless
// --profile names one to keep.
//   node scripts/chrome-gate.mjs <file.gguf> <refs.json> --models <dir> [--slots 8 | --budget GB] [--draft <drafter.gguf>] [--lora <adapter.gguf> [--lora-scale 1]] [--extra <side file,…>] [--dist <dir>] [--env K=V,…] [--log chrome.log] [--profile dir] [--keep] [--out x.json]
// --draft: a drafter GGUF in --models for speculative decoding (DFlash, DSpark, MTP); replies must stay identical.
// --lora: a LoRA adapter GGUF in --models, applied at --lora-scale (default 1) as llama-server --lora does.
// --extra: side files in --models that the model opens beside the GGUF (DeepSeek-V4.1's Engram file).
// --dist: serve ggml/dist/ from this folder instead, to gate a browser build of another fork branch.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, realpathSync, openSync } from 'node:fs';
import { readFile, stat, realpath, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, extname, normalize, sep } from 'node:path';

const [file, refs] = process.argv.slice(2);
const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
if (!file || !refs || !arg('models')) { console.error('usage: node scripts/chrome-gate.mjs <file.gguf> <refs.json> --models <dir> [--slots 8] [--keep] [--out x.json]'); process.exit(2); }
const root = realpathSync(new URL('..', import.meta.url).pathname);
const models = realpathSync(arg('models'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };

async function inside(base, rel) {
  try { const p = await realpath(join(base, rel)); return p.startsWith(base + sep) && (await stat(p)).isFile() ? p : null; } catch { return null; }
}
async function linkInside(base, rel) {
  const p = join(base, rel);
  if (!p.startsWith(base + sep) || p.slice(base.length + 1).includes(sep)) return null;
  try { return (await stat(p)).isFile() ? p : null; } catch { return null; }
}
const server = createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const fromModels = path.startsWith('/models/');
    // --models may hold links to the files elsewhere (gated GGUFs move to an external disk): the name must be inside
    // the folder, its target need not be
    const dist = arg('dist') && path.startsWith('/ggml/dist/') ? realpathSync(arg('dist')) : null;
    const p = fromModels ? await linkInside(models, path.slice(8)) : dist ? await inside(dist, path.slice(11)) : await inside(root, path);
    if (!p) { res.writeHead(404).end(); return; }
    if (!fromModels) { res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-store' }).end(await readFile(p)); return; }
    const size = (await stat(p)).size;
    const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
    const start = m ? Number(m[1]) : 0, end = m ? Math.min(size - 1, m[2] ? Number(m[2]) : size - 1) : size - 1;
    res.writeHead(m ? 206 : 200, { 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes', ...(m ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}) });
    createReadStream(p, { start, end }).on('error', () => res.destroy()).pipe(res);
  } catch { if (!res.headersSent) res.writeHead(500); res.end(); }
}).listen(Number(arg('port', 8192)), '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port, cdp = 9500 + (port % 400);
const own = !arg('profile');
const profile = arg('profile') || await mkdtemp(join(homedir(), '.cache/diskformer-gate-'));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdp}`,
    ...(arg('log') ? ['--enable-logging=stderr', '--v=0'] : []), 'about:blank'],
  // --log <file>: Chrome's log, with the worker's console and any wasm abort, for a gate that stalls
  { stdio: arg('log') ? ['ignore', 'ignore', openSync(arg('log'), 'w')] : 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Chrome's helper processes write into the profile for a moment after the main process exits: retry the delete
const dropProfile = async () => {
  if (!own) return;
  for (let k = 0; k < 10; k++) {
    try { await rm(profile, { recursive: true, force: true }); return; } catch { await sleep(1000); }
  }
  console.error(`could not delete the gate profile ${profile}`);
};
// stopped from outside (timeout, kill): take Chrome and its throwaway profile down too, so no copy is left behind
for (const sig of ['SIGTERM', 'SIGINT']) process.on(sig, async () => {
  try { chrome.kill(); } catch {}
  await new Promise((r) => { chrome.once('exit', r); setTimeout(r, 5000); });
  await dropProfile();
  process.exit(3);
});
let ws;
for (let k = 0; k < 100 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdp}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
await new Promise((r) => { ws.onopen = r; });
let id = 0; const pend = new Map(); ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }); return r.result && r.result.result && r.result.result.value; };

// a split GGUF (name-00001-of-0000n.gguf): every part is copied, and the gate loads the first
const split = /^(.*)-00001-of-(\d{5})\.gguf$/.exec(file);
const parts = split ? Array.from({ length: +split[2] }, (_, k) => `${split[1]}-${String(k + 1).padStart(5, '0')}-of-${split[2]}.gguf`) : [file];
const sizes = await Promise.all(parts.map(async (f) => (await stat(join(models, f))).size));
// --budget <GB> plans slots (and the batch size) from a GPU budget, as the demo page does; else --slots, default 8
const sizing = arg('budget') ? { budget: arg('budget') } : { slots: arg('slots', '8') };
// --env NAME=value,NAME=value: driver settings (DF_UBATCH, DF_OUTPUT_CPU, …), as gate.html takes them
const q = new URLSearchParams({ model: file, refs: '/' + refs.replace(/^\//, ''), ...sizing, url: parts.map((f) => `/models/${f}`).join(','), size: sizes.join(','), ...(arg('env') ? { env: arg('env') } : {}), ...(arg('draft') ? { draft_url: `/models/${arg('draft')}`, draft_size: String((await stat(join(models, arg('draft')))).size) } : {}),
  ...(arg('lora') ? { lora_url: `/models/${arg('lora')}`, lora_size: String((await stat(join(models, arg('lora')))).size), ...(arg('lora-scale') ? { lora_scale: arg('lora-scale') } : {}) } : {}),
  ...(arg('extra') ? { extra_url: arg('extra').split(',').map((f) => `/models/${f}`).join(','), extra_size: (await Promise.all(arg('extra').split(',').map(async (f) => (await stat(join(models, f))).size))).join(',') } : {}) });
await send('Page.navigate', { url: `http://127.0.0.1:${port}/ggml/gate.html?${q}` });
let gate = null, last = '', lastAt = Date.now();
const stall = Number(arg('stall', 10)) * 60e3;   // no new line on the page for this long: the gate is stuck
for (const t0 = Date.now(); Date.now() - t0 < Number(arg('minutes', 120)) * 60e3 && !gate; await sleep(2000)) {
  gate = await ev('window.gate || null');
  const out = await ev('(document.getElementById("out") || {}).textContent || ""');
  const line = out.split('\n').pop();
  if (line && line !== last) { console.error(`${new Date().toTimeString().slice(0, 8)} ${line}`); last = line; lastAt = Date.now(); }
  if (!gate && Date.now() - lastAt > stall) { gate = { error: `stalled: no progress for ${stall / 60e3} min after "${last}"` }; console.error(gate.error); }
}
const result = gate ? { file, refs, ...sizing, same: gate.same, total: gate.total, error: gate.error,
  differ: (gate.results || []).filter((r) => !r.ok).map((r) => r.name), texts: (gate.results || []).map((r) => ({ name: r.name, ok: r.ok, text: r.text })) } : { file, refs, error: 'timed out' };
if (arg('out')) await writeFile(arg('out'), JSON.stringify(result, null, 1));
console.log(JSON.stringify({ ...result, texts: undefined }));
try { chrome.kill(); } catch {}
await new Promise((r) => { if (chrome.exitCode !== null) r(); else { chrome.once('exit', r); setTimeout(r, 10000); } });
await dropProfile();
server.close();
process.exit(result.error ? 2 : 0);
