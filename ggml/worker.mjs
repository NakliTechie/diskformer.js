// Runs df-chat.wasm (llama.cpp + WebGPU_Paged) in a worker. The GGUF lives in OPFS (the origin's private disk), under
// gguf/<name>; llama.cpp loads it through a WORKERFS mount of that file, and experts are read on demand through OPFS
// sync access handles, straight into the wasm heap. A GGUF gets into OPFS by download (resumable, from a URL) or by
// copying a picked file; with opfs: false a picked File is read in place instead (slower).
//
// Messages in: load {opfsName | file, budget | slots, ctx, env} · chat {messages, n} · complete {prompt, n} (the gate)
//   · abort · download {url, name, size} · list · remove {name}
// Messages out: log · progress · plan · loaded · piece · done · downloaded · models · removed · error
import createModule from './dist/df-chat.mjs';  // built from ggml/build-wasm, see ggml/dist/BUILD.md
import { OpfsReaderPool } from '../src/opfs-reader.js';

let mod;
const post = (type, data) => postMessage({ type, ...data });
// memory64: raw exports take pointers as BigInt and return them as BigInt
const str = (s) => BigInt(mod.stringToNewUTF8(s));
const text = (p) => mod.UTF8ToString(Number(p));
const ggufDir = async (create = true) => (await navigator.storage.getDirectory()).getDirectoryHandle('gguf', { create });

// The picked file in OPFS under its name, copied unless a file of the same name and size is already there.
async function opfsCopy(file) {
  const dir = await ggufDir();
  const fh = await dir.getFileHandle(file.name, { create: true });
  const existing = await fh.getFile();
  if (existing.size !== file.size) {
    const t0 = performance.now();
    const handle = await fh.createSyncAccessHandle();
    handle.truncate(0);
    let at = 0;
    const reader = file.stream().getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      handle.write(value, { at });
      at += value.byteLength;
      if ((at & 0x3fffffff) < value.byteLength) post('progress', { name: file.name, copied: at, total: file.size });
    }
    handle.flush();
    handle.close();
    post('log', { line: `df: copied ${file.name} into OPFS in ${((performance.now() - t0) / 1000).toFixed(1)} s` });
  }
  return fh;
}

// A GGUF from a URL into OPFS as <name>.part, in 64 MiB chunks over STREAMS parallel Range requests (one stream from
// Hugging Face gave ~6 MB/s here, eight gave ~25), renamed to <name> once every chunk is written. <name>.part.json
// lists the finished chunks, so a stopped download resumes without gaps. A chunk counts as finished only after its
// bytes are flushed. A .part with no usable list starts over: with parallel streams its length is the furthest
// write, not the finished bytes (a reload before the first chunk finished once left 5 chunks half written).
const CHUNK = 64 << 20, STREAMS = 6;
let downloading = null;
async function download({ url, name, size }) {
  const dir = await ggufDir();
  const done = await dir.getFileHandle(name).then((h) => h.getFile()).catch(() => null);
  if (done && done.size === size) { post('downloaded', { name, size, ms: 0 }); return; }
  const fh = await dir.getFileHandle(`${name}.part`, { create: true });
  const handle = await fh.createSyncAccessHandle();
  const sideFh = await dir.getFileHandle(`${name}.part.json`, { create: true });
  const side = await sideFh.createSyncAccessHandle();
  const n = Math.ceil(size / CHUNK), chunkLen = (i) => Math.min(size, (i + 1) * CHUNK) - i * CHUNK;
  let finished = new Set();
  try {
    const buf = new Uint8Array(side.getSize()); side.read(buf, { at: 0 });
    const j = JSON.parse(new TextDecoder().decode(buf) || 'null');
    if (j && j.size === size && j.chunk === CHUNK) finished = new Set(j.done);
  } catch {}
  if (!finished.size || handle.getSize() > size) { handle.truncate(0); finished.clear(); }
  const save = () => {
    const b = new TextEncoder().encode(JSON.stringify({ size, chunk: CHUNK, done: [...finished] }));
    side.truncate(0); side.write(b, { at: 0 }); side.flush();
  };
  const ctrl = new AbortController();
  downloading = ctrl;
  const t0 = performance.now();
  const start = [...finished].reduce((a, i) => a + chunkLen(i), 0);
  let got = start, last = 0;
  const queue = [...Array(n).keys()].filter((i) => !finished.has(i));
  const stream = async () => {
    for (let i; (i = queue.shift()) !== undefined;) {
      for (let attempt = 1; ; attempt++) {
        let wrote = 0;
        try {
          const a = i * CHUNK, b = a + chunkLen(i) - 1;
          const r = await fetch(url, { headers: { Range: `bytes=${a}-${b}` }, signal: ctrl.signal });
          if (r.status !== 206) throw new Error(`download: HTTP ${r.status}`);
          const reader = r.body.getReader();
          for (;;) {
            const { done: eof, value } = await reader.read();
            if (eof) break;
            handle.write(value, { at: a + wrote });
            wrote += value.byteLength; got += value.byteLength;
            const now = performance.now();
            if (now - last > 500) { last = now; post('progress', { name, copied: got, total: size, rate: (got - start) / ((now - t0) / 1000) }); }
          }
          if (wrote !== chunkLen(i)) throw new Error(`download: chunk ${i} ended at ${wrote} of ${chunkLen(i)} bytes`);
          handle.flush(); finished.add(i); save();
          break;
        } catch (e) {
          got -= wrote;
          if (ctrl.signal.aborted || attempt >= 5) throw e;
          await new Promise((res) => setTimeout(res, 1000 * attempt));
        }
      }
    }
  };
  try {
    await Promise.all(Array.from({ length: STREAMS }, stream));
    if (finished.size !== n) throw new Error(`download: ${finished.size} of ${n} chunks`);
    handle.truncate(size); handle.flush(); handle.close(); side.close();
    await dir.removeEntry(`${name}.part.json`);
    await fh.move(name);
    post('downloaded', { name, size, ms: Math.round(performance.now() - t0) });
  } catch (e) {
    ctrl.abort();
    try { handle.flush(); handle.close(); } catch {}
    try { save(); side.close(); } catch {}
    throw ctrl.signal.aborted && !/HTTP|chunk/.test(e.message) ? new Error(`download of ${name} stopped at ${(got / 1e9).toFixed(2)} GB; it resumes from there`) : e;
  } finally {
    downloading = null;
  }
}

// Every GGUF in OPFS: complete files and partial downloads.
async function list() {
  const dir = await ggufDir();
  const models = [];
  for await (const [name, h] of dir.entries()) {
    if (h.kind !== 'file' || name.endsWith('.part.json') || name.endsWith('.crswap')) continue;  // .crswap: Chrome's temp copy during a write
    const f = await h.getFile();
    models.push({ name: name.replace(/\.part$/, ''), size: f.size, partial: name.endsWith('.part') });
  }
  return models.sort((a, b) => a.name.localeCompare(b.name));
}

onmessage = async ({ data }) => {
  try {
    if (data.type === 'abort') {
      if (downloading) downloading.abort();
      if (mod) mod._df_abort();
    } else if (data.type === 'list') {
      post('models', { models: await list() });
    } else if (data.type === 'remove') {
      const dir = await ggufDir();
      for (const n of [data.name, `${data.name}.part`, `${data.name}.part.json`]) await dir.removeEntry(n).catch(() => {});
      post('removed', { name: data.name, models: await list() });
    } else if (data.type === 'download') {
      await download(data);
      post('models', { models: await list() });
    } else if (data.type === 'load') {
      mod = await createModule({
        print: (line) => post('log', { line }),
        printErr: (line) => post('log', { line }),
        onPiece: (piece) => post('piece', { piece }),
        // settings for llama.cpp / ggml-webgpu, e.g. { GGML_WEBGPU_PAGED_STATS: '2' }
        preRun: [(m) => Object.assign(m.ENV, data.env || {})],
      });
      let file = data.file;
      let fh = null;
      if (!file) {
        fh = await (await ggufDir(false)).getFileHandle(data.opfsName);  // a GGUF already in OPFS, by name
      } else if (data.opfs !== false) {
        fh = await opfsCopy(data.file);
      }
      if (fh) {
        file = await fh.getFile();
        // read-only, so the reader workers can open the same file
        mod.ggufHandle = await fh.createSyncAccessHandle({ mode: 'read-only' });
        // expert reads in parallel, off this thread (DF_READERS=0: on this thread's handle)
        const n = +((data.env || {}).DF_READERS ?? 8);  // 8: measured 10.66 tok/s vs 9.72 at 4, 10.57 at 16
        if (n > 0) mod.ggufReaders = await OpfsReaderPool.open(`gguf/${file.name}`, { workers: n });
      }
      mod.ggufFile = file;
      mod.FS.mkdir('/m');
      mod.FS.mount(mod.WORKERFS, { files: [file] }, '/m');
      const t0 = performance.now();
      // a GPU budget (bytes) becomes slots through df_plan; else data.slots as given
      let slots = data.slots, plan = null;
      if (data.budget) {
        plan = JSON.parse(text(await mod._df_plan(str('/m/' + file.name), data.budget, data.ctx ?? 1024)));
        post('plan', { plan });
        if (plan.error) throw new Error(`budget ${(data.budget / 1e9).toFixed(1)} GB: ${plan.error}`);
        slots = plan.slots;
      }
      const rc = await mod._df_load(str('/m/' + file.name), slots, data.ctx ?? 1024);
      post('loaded', { rc, slots, plan, size: file.size, name: file.name, ms: Math.round(performance.now() - t0) });
    } else if (data.type === 'complete') {
      const reply = await mod._df_complete(str(data.prompt), data.n ?? 64);
      post('done', { id: data.id, text: text(reply), stats: JSON.parse(text(mod._df_stats())) });
    } else if (data.type === 'chat') {
      // messages: the whole conversation; prompt: one user turn (the older single-turn form)
      const reply = data.messages
        ? await mod._df_chat_messages(str(JSON.stringify(data.messages)), data.n ?? 512)
        : await mod._df_chat(str(data.prompt), data.n ?? 64);
      post('done', { text: text(reply), stats: JSON.parse(text(mod._df_stats())) });
    }
  } catch (e) {
    // a C++ exception from the wasm arrives as a WebAssembly.Exception: report its message, not "[object ...]"
    let message = String(e?.message ?? e);
    if (typeof WebAssembly.Exception === 'function' && e instanceof WebAssembly.Exception && mod?.getExceptionMessage) {
      try { message = `C++ exception: ${mod.getExceptionMessage(e).filter(Boolean).join(': ')}`; } catch {}
    }
    post('error', { message, stack: String(e?.stack ?? '') });
  }
};
