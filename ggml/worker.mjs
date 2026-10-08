// Runs df-chat.wasm (llama.cpp + WebGPU_Paged) in a worker. The GGUF is copied once into OPFS (the origin's private
// disk); llama.cpp loads it through a WORKERFS mount of that file, and experts are read on demand through an OPFS sync
// access handle, straight into the wasm heap. With opfs: false the picked File is read in place instead (slower).
import createModule from './build-wasm/df-chat.mjs';
import { OpfsReaderPool } from '../src/opfs-reader.js';

let mod;
const post = (type, data) => postMessage({ type, ...data });
// memory64: raw exports take pointers as BigInt and return them as BigInt
const str = (s) => BigInt(mod.stringToNewUTF8(s));
const text = (p) => mod.UTF8ToString(Number(p));

// The picked file in OPFS under its name, copied unless a file of the same name and size is already there.
async function opfsCopy(file) {
  const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('gguf', { create: true });
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
      if ((at & 0x3fffffff) < value.byteLength) post('progress', { copied: at, total: file.size });
    }
    handle.flush();
    handle.close();
    post('log', { line: `df: copied ${file.name} into OPFS in ${((performance.now() - t0) / 1000).toFixed(1)} s` });
  }
  return fh;
}

onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
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
        // a GGUF already in OPFS, by name
        const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('gguf');
        fh = await dir.getFileHandle(data.opfsName);
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
      post('loaded', { rc, slots, plan, ms: Math.round(performance.now() - t0) });
    } else if (data.type === 'complete') {
      const reply = await mod._df_complete(str(data.prompt), data.n ?? 64);
      post('done', { id: data.id, text: text(reply), stats: JSON.parse(text(mod._df_stats())) });
    } else if (data.type === 'chat') {
      const reply = await mod._df_chat(str(data.prompt), data.n ?? 64);
      post('done', { text: text(reply), stats: JSON.parse(text(mod._df_stats())) });
    }
  } catch (e) {
    post('error', { message: String(e?.stack ?? e) });
  }
};
