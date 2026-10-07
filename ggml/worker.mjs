// Runs df-chat.wasm (llama.cpp + WebGPU_Paged) in a worker. The GGUF is a WORKERFS mount of the File the page sends:
// read in place, never copied; experts are read from it on demand.
import createModule from './build-wasm/df-chat.mjs';

let mod;
const post = (type, data) => postMessage({ type, ...data });
// memory64: raw exports take pointers as BigInt and return them as BigInt
const str = (s) => BigInt(mod.stringToNewUTF8(s));
const text = (p) => mod.UTF8ToString(Number(p));

onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      mod = await createModule({
        print: (line) => post('log', { line }),
        printErr: (line) => post('log', { line }),
        onPiece: (piece) => post('piece', { piece }),
      });
      mod.FS.mkdir('/m');
      mod.FS.mount(mod.WORKERFS, { files: [data.file] }, '/m');
      const t0 = performance.now();
      const rc = await mod._df_load(str('/m/' + data.file.name), data.slots, data.ctx ?? 1024);
      post('loaded', { rc, ms: Math.round(performance.now() - t0) });
    } else if (data.type === 'chat') {
      const reply = await mod._df_chat(str(data.prompt), data.n ?? 64);
      post('done', { text: text(reply), stats: JSON.parse(text(mod._df_stats())) });
    }
  } catch (e) {
    post('error', { message: String(e?.stack ?? e) });
  }
};
