// Runs an example in headless Chrome and prints window.result. Serves the repo itself on a free port.
//   node examples/run-demo.mjs [rows|embedding] [--profile dir]
//   node examples/run-demo.mjs test/browser/<page>.html [--models dir --file name.gguf]   a browser test page
//   node examples/run-demo.mjs chat [--model gemma|qwen] [--budget GB] [--ctx N] [--models dir] [--root name]
//                                   [--profile dir] [--port N]   (defaults: ~/.cache/diskformer-chat-profile, 8191)
// chat replays examples/chat/refs/<model>.json (llama.cpp's replies) under a GPU budget and fails on any reply that
// differs. With --models, the GGUF is served from that folder (HTTP Range) instead of downloaded from Hugging Face.
// OPFS belongs to the origin, so chat's fixed default profile and port store the model once across runs.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, realpathSync } from 'node:fs';
import { readFile, stat, realpath, mkdtemp } from 'node:fs/promises';
import { tmpdir, homedir } from 'node:os';
import { join, extname, normalize, sep } from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const root = realpathSync(new URL('..', import.meta.url).pathname);
const page = (process.argv[2] || '').endsWith('.html') ? process.argv[2].replace(/^\.?\//, '') : null;
const example = ['rows', 'embedding', 'chat'].includes(process.argv[2]) ? process.argv[2] : 'rows';
const chat = !page && example === 'chat';
const models = arg('models') && realpathSync(arg('models'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };

// The real path of base/rel when it is a file inside base (symlinks resolved), else null.
async function inside(base, rel) {
  try {
    const p = await realpath(join(base, rel));
    return (p.startsWith(base + sep) && (await stat(p)).isFile()) ? p : null;
  } catch { return null; }
}
const server = createServer(async (req, res) => {
  try {
    let path;
    try { path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)); } catch { res.writeHead(400).end(); return; }
    if (models && path.startsWith('/models/')) {        // a GGUF, read in byte ranges
      const p = await inside(models, path.slice('/models/'.length));
      if (!p) { res.writeHead(404).end(); return; }
      const size = (await stat(p)).size;
      const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
      let start = 0, end = size - 1;
      if (m) {
        start = Number(m[1]); end = Math.min(size - 1, m[2] ? Number(m[2]) : size - 1);
        if (start >= size || end < start) { res.writeHead(416, { 'Content-Range': `bytes */${size}` }).end(); return; }
      }
      res.writeHead(m ? 206 : 200, { 'Content-Length': size ? end - start + 1 : 0, 'Accept-Ranges': 'bytes', ...(m ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}) });
      if (!size) { res.end(); return; }
      createReadStream(p, { start, end }).on('error', () => res.destroy()).pipe(res);
      return;
    }
    const p = await inside(root, path);
    if (!p) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' }).end(await readFile(p));
  } catch { if (!res.headersSent) res.writeHead(500); res.end(); }
}).listen(Number(arg('port', chat ? 8191 : 0)), '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port, cdp = 9500 + (port % 400);
// chat keeps one profile and port by default, so OPFS (per origin) holds the model across runs.
const profile = arg('profile') || (chat ? join(homedir(), '.cache/diskformer-chat-profile') : await mkdtemp(join(tmpdir(), 'diskformer-demo-')));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdp}`, 'about:blank'], { stdio: 'ignore' });
const stop = (code) => { try { chrome.kill(); } catch {} server.close(); process.exit(code); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdp}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
await new Promise((r) => { ws.onopen = r; });
let id = 0; const pend = new Map(); ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, returnByValue: true }); return r.result && r.result.result && r.result.result.value; };

let url = page ? `http://127.0.0.1:${port}/${page}` : `http://127.0.0.1:${port}/examples/${example}/index.html`, minutes = page ? 30 : 3;
if (page && models && arg('file')) url += `?url=/models/${encodeURIComponent(arg('file'))}`;
if (chat) {
  const model = arg('model', 'gemma');
  const file = { gemma: 'gemma-4-26B_q4_0-it.gguf', qwen: 'Qwen3.6-35B-A3B-Q8_0.gguf' }[model];
  const p = new URLSearchParams({ model, gate: `refs/${model}.json`, ctx: arg('ctx', '1024') });
  if (arg('budget')) p.set('budget', arg('budget'));
  if (arg('root')) p.set('root', arg('root'));
  if (models) p.set('url', `/models/${file}`);
  url += `?${p}`; minutes = 90;                         // a first run downloads the model
}
await send('Page.navigate', { url });
let last = '', lastAt = 0;
for (const t0 = Date.now(); Date.now() - t0 < minutes * 60e3; await sleep(1000)) {
  const v = await ev('window.result || null');
  if (v) { console.log(JSON.stringify(v, null, 1)); stop(v.ok ? 0 : 1); }
  const p = await ev('window.progressText || ""');            // a new step at once, the same step every 30 s
  if (p && p !== last && (p.slice(0, 24) !== last.slice(0, 24) || Date.now() - lastAt > 30e3)) { console.error(`${new Date().toTimeString().slice(0, 8)} ${p}`); last = p; lastAt = Date.now(); }
}
console.error('demo timed out'); stop(2);
