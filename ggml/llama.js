// diskformer/llama: llama.cpp with WebGPU in a worker, MoE experts paged from OPFS (the fork NakliTechie/llama.cpp,
// branch webgpu-diskformer; build in ggml/dist/BUILD.md). One request at a time per Llama, as the worker runs them.
//
//   import { Llama } from 'diskformer/llama';
//   const llama = new Llama();
//   await llama.download({ url, name: 'model.gguf', size }, (p) => console.log(p.copied / p.total));
//   const info = await llama.load({ name: 'model.gguf', budget: 4e9 });   // or slots: 8
//   const { text, stats } = await llama.chat([{ role: 'user', content: 'Hi' }], { n: 256, onPiece: (s) => {} });
//
// The worker's messages are documented at the top of ggml/worker.mjs; this file only gives them promises.

// The worker script, started from a blob when it lives on another origin (a CDN): browsers refuse a cross-origin
// worker URL but let a module worker import a cross-origin module that allows CORS.
function startWorker() {
  const url = new URL('./worker.mjs', import.meta.url);
  if (url.origin === location.origin) return new Worker(url, { type: 'module' });
  const stub = URL.createObjectURL(new Blob([`import ${JSON.stringify(url.href)};`], { type: 'text/javascript' }));
  return new Worker(stub, { type: 'module' });
}

export class Llama {
  constructor({ worker } = {}) {
    this.worker = worker || startWorker();
    this.busy = Promise.resolve();
    this.onLog = null;  // (line) => {}: llama.cpp's log
  }

  // Sends one message and resolves on the first of `until`; `onEvent` sees every other message meanwhile.
  request(msg, until, onEvent) {
    const run = () => new Promise((resolve, reject) => {
      this.worker.onmessage = ({ data }) => {
        if (data.type === 'log') { if (this.onLog) this.onLog(data.line); return; }
        if (data.type === 'error') { reject(new Error(data.message)); return; }
        if (until.includes(data.type)) { resolve(data); return; }
        if (onEvent) onEvent(data);
      };
      this.worker.postMessage(msg);
    });
    const p = this.busy.then(run, run);
    this.busy = p.catch(() => {});
    return p;
  }

  // GGUFs in this origin's OPFS: [{ name, size, partial }]
  async list() { return (await this.request({ type: 'list' }, ['models'])).models; }
  async remove(name) { return (await this.request({ type: 'remove', name }, ['removed'])).models; }

  // A GGUF from a URL (Range requests) into OPFS, resumable; onProgress({ copied, total, rate }).
  download({ url, name, size }, onProgress) {
    return this.request({ type: 'download', url, name, size }, ['models'], (d) => { if (d.type === 'progress' && onProgress) onProgress(d); });
  }

  // Loads a stored GGUF: budget (bytes of GPU memory, slots planned from it) or slots (experts per layer on the GPU).
  // draft: a stored drafter GGUF (DFlash, DSpark, MTP...) for speculative decoding; replies stay greedy-identical.
  // extra: stored side files the model opens beside the GGUF, by name (DeepSeek-V4.1's Engram tables).
  load({ name, budget = 0, slots = 0, ctx = 1024, env = {}, draft = null, extra = [] }, onPlan) {
    return this.request({ type: 'load', opfsName: name, budget, slots, ctx, env, draft, extra }, ['loaded'], (d) => { if (d.type === 'plan' && onPlan) onPlan(d.plan); });
  }

  // A reply through the model's chat template; greedy. Resolves { text, stats }. thinking: false renders the template
  // with reasoning off (models that reason first then answer directly); default: the template's own choice.
  chat(messages, { n = 512, thinking = true, onPiece } = {}) {
    return this.request({ type: 'chat', messages, n, thinking }, ['done'], (d) => { if (d.type === 'piece' && onPiece) onPiece(d.piece); });
  }

  // A continuation of an already formatted prompt; greedy.
  complete(prompt, { n = 64, onPiece } = {}) {
    return this.request({ type: 'complete', prompt, n }, ['done'], (d) => { if (d.type === 'piece' && onPiece) onPiece(d.piece); });
  }

  // Stops the reply or download in progress (not queued behind it).
  abort() { this.worker.postMessage({ type: 'abort' }); }

  terminate() { this.worker.terminate(); }
}
