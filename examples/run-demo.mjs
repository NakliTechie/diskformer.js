// Runs examples/rows in headless Chrome and prints window.result. Serves the repo itself on a free port.
//   node examples/run-demo.mjs [--profile dir]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, extname, normalize } from 'node:path';
const root = new URL('..', import.meta.url).pathname;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript' };
const server = createServer(async (req, res) => {
  const p = join(root, normalize(new URL(req.url, 'http://x').pathname));
  let body;
  try { body = await readFile(p); } catch { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'Content-Type': types[extname(p)] || 'application/octet-stream' }).end(body);
}).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port, cdp = 9500 + (port % 400);
const i = process.argv.indexOf('--profile');
const profile = i > 0 ? process.argv[i + 1] : await mkdtemp(join(tmpdir(), 'diskformer-demo-'));
const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ['--headless=new', '--no-first-run', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--remote-debugging-port=${cdp}`, 'about:blank'], { stdio: 'ignore' });
const stop = (code) => { try { chrome.kill(); } catch {} server.close(); process.exit(code); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let k = 0; k < 100 && !ws; k++) { try { const t = (await (await fetch(`http://127.0.0.1:${cdp}/json`)).json()).find((x) => x.type === 'page'); ws = new WebSocket(t.webSocketDebuggerUrl); } catch { await sleep(200); } }
await new Promise((r) => { ws.onopen = r; });
let id = 0; const pend = new Map(); ws.onmessage = (e) => { const m = JSON.parse(e.data); if (pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const n = ++id; pend.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
await send('Page.navigate', { url: `http://127.0.0.1:${port}/examples/rows/index.html` });
for (let k = 0; k < 600; k++) {
  const r = await send('Runtime.evaluate', { expression: 'window.result || null', returnByValue: true });
  const v = r.result && r.result.result && r.result.result.value;
  if (v) { console.log(JSON.stringify(v, null, 1)); stop(v.ok ? 0 : 1); }
  await sleep(250);
}
console.error('demo timed out'); stop(2);
