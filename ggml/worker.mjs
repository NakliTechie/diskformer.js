// Runs df-chat.wasm (llama.cpp + WebGPU_Paged) in a worker. The GGUF is copied once into OPFS (the origin's private
// disk); llama.cpp loads it through a WORKERFS mount of that file, and experts are read on demand through an OPFS sync
// access handle, straight into the wasm heap. With opfs: false the picked File is read in place instead (slower).
import createModule from './build-wasm/df-chat.mjs';

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
      });
      let file = data.file;
      if (!file) {
        // a GGUF already in OPFS, by name
        const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('gguf');
        const fh = await dir.getFileHandle(data.opfsName);
        file = await fh.getFile();
        mod.ggufHandle = await fh.createSyncAccessHandle();
      } else if (data.opfs !== false) {
        const fh = await opfsCopy(data.file);
        file = await fh.getFile();
        mod.ggufHandle = await fh.createSyncAccessHandle();
      }
      mod.ggufFile = file;
      mod.FS.mkdir('/m');
      mod.FS.mount(mod.WORKERFS, { files: [file] }, '/m');
      const t0 = performance.now();
      const rc = await mod._df_load(str('/m/' + file.name), data.slots, data.ctx ?? 1024);
      post('loaded', { rc, ms: Math.round(performance.now() - t0) });
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
