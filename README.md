<h1 align="center">diskformer.js</h1>

**Run a model larger than your GPU memory in a browser tab.** The weights stay on disk in the browser's private
storage (OPFS); WebGPU holds only what each step needs. A disk tier for in-browser inference, with two reference
engines that use it.

Plain ES modules, no dependencies, no build step. Chromium with WebGPU. Weights stay in the user's own browser storage.

[![npm](https://img.shields.io/npm/v/diskformer?style=flat-square&color=555)](https://www.npmjs.com/package/diskformer)
[![license](https://img.shields.io/badge/license-MIT-555?style=flat-square)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-555?style=flat-square)](package.json)

## Install

| How | Command |
|---|---|
| npm | `npm install diskformer` |
| CDN | `import { Gemma4MoeSsd } from 'https://cdn.jsdelivr.net/npm/diskformer@0.2/engines/gemma4_moe_ssd.js'` |
| Copy | `src/`, `engines/` and `index.js`; nothing else is needed |

## Try it: a 14.4 GB model in 2.5 GB of GPU memory

`examples/chat` is one static page. Pick a model and a GPU memory budget. The page downloads the model once into
OPFS, resumably, then chats. Serve the repo and open it:

```bash
python3 -m http.server 8000      # then open http://localhost:8000/examples/chat/
```

Measured on a MacBook M4 Pro with 24 GB, Chrome 154, on 2026-10-05:

| Model | On disk | GPU budget | GPU memory used | Decode | Output |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B, QAT Q4_0 | 14.4 GB | 2.5 GB | 2.02 GB | 12.4 tok/s | 9/9 replies identical to llama.cpp |
| Qwen3.6 35B-A3B, Q8_0 | 36.9 GB | 4 GB | 3.52 GB | 9.2 tok/s | not compared |

"Identical" means character for character, against llama.cpp b9830's Metal path on the same GGUF. The test covers
9 conversations, greedy decoding, up to 64 tokens each. Qwen3.6 has no llama.cpp comparison: its Metal path cannot
hold 36.9 GB on this Mac.

A larger budget is faster. In the engine benchmark, Gemma decodes at 13.3 tok/s with 2.5 GB and 18.3 with 4.3 GB.
With a 4 GiB expert cache it decodes at 23.6, in a 6.9 GB GPU process. Only Apple silicon is tested; discrete GPUs
and 8–16 GB machines are not.

In your own page:

```js
import { Gemma4MoeSsd } from 'diskformer/engines/gemma4_moe_ssd.js';

const engine = await Gemma4MoeSsd.load(null, { gpuBudgetBytes: 3e9, onProgress: console.log });
for await (const { text } of engine.generate([{ role: 'user', content: 'Hello' }], { maxNewTokens: 128 })) show(text);
```

The first load downloads the GGUF from Hugging Face and copies it into OPFS in the engine's layout. Later loads take
seconds.

## Why

Your model does not fit on the GPU, or it fits only by crowding out everything else. Browser engines load every
weight onto the GPU at load time, including the ones a step never touches. A mixture-of-experts model uses under a
tenth of its weights per token. Gemma 4's per-layer embedding table is read one row per token.

diskformer keeps those weights in OPFS, read through sync access handles in workers. Only what each step needs
moves into a GPU cache. The core is the store and the residency layer: your engine keeps its own kernels and decides
what a step needs. `engines/` shows two complete engines built on it.

## The engines

`engines/gemma4_moe_ssd.js` runs Gemma 4 26B-A4B. `engines/qwen35_moe_ssd.js` runs Qwen3.6 35B-A3B, and
`engines/qwen3_moe_ssd.js` runs Qwen3-30B-A3B as their base. They are from-scratch WebGPU engines. The dense layers and
KV cache live on the GPU. The experts stream from OPFS into a GPU cache, sized by `gpuBudgetBytes`. A budget that
cannot hold the dense part and two tokens' experts is refused with the numbers. Decoding is greedy. The kernels are
ported from gemma4-webgpu (Apache-2.0, `engines/NOTICE`). The engine files are LocalMind's, kept in step by
`scripts/sync-localmind.mjs`.

## A table on disk

`RowFile` holds fixed-size rows in one OPFS file with a manifest, written once and validated by a fingerprint.
`RowCache` keeps `slots` rows on the GPU, split into planes when a row's parts live in different buffers, with an
O(1) LRU and an optional GPU id→slot map that kernels can read. `examples/embedding` is the whole loop in one page.

## Experts on disk

`ingestGguf({ url, key, plan })` streams a GGUF over HTTP ranges into OPFS in your engine's layout. `plan.layout` says
where each tensor goes. `plan.units` lists the byte ranges to copy or split: Q8_0 and Q4_0 blocks become a value plane
and an f16-scale plane, bit-exact. An interrupted ingest resumes from its last 1 GiB checkpoint. `ExpertStreamer`
(also `RecordPool`) keeps a fixed GPU slot pool of expert records. `ensure(layer, ids)` returns pinned slots,
`prefetch(layer, guesses)` never blocks, and uploads go through a mapped staging ring.

## Measure the disk

`measure({ path, recordBytes })` reports sequential MB/s and random whole-record reads, one at a time and in bursts
of 8, through the reader pool. OPFS speed varies a lot between machines; measure where your users run.

## Commands

```bash
npm test                                     # Node unit tests: record pool, row cache, GGUF parse and splits, ingest helpers
node examples/run-demo.mjs rows              # headless Chrome: a 64 MB table on disk, 2,048 rows on the GPU, every byte checked
node examples/run-demo.mjs embedding         # lookup → slots → a WGSL gather kernel, every output value checked
node examples/run-demo.mjs chat --budget 2.5 # downloads Gemma once (14.4 GB), replays 9 llama.cpp replies, fails on any difference
node examples/run-demo.mjs test/browser/ingest-resume.html --models <dir> --file <gguf>   # abort, resume, compare bytes
npm run sync                                 # the files shared with LocalMind match their sources
```

## Verify it yourself

`node examples/run-demo.mjs chat` runs the chat page in headless Chrome under a GPU budget. It replays the
conversations in `examples/chat/refs/gemma.json`, which holds llama.cpp's replies, and exits 1 on any reply that
differs. Pass `--models <dir>` to serve a local GGUF instead of downloading it. The resume test aborts an ingest at
60%, resumes it, and compares every byte with a clean ingest.

The code runs in [LocalMind](https://github.com/NakliTechie/LocalMind) on its live site. `src/` (all but `measure.js`)
and `engines/` are byte-identical to LocalMind's copies, apart from the engines' import paths.

## License

MIT. `engines/` contains kernels ported from Apache-2.0 code: see `engines/NOTICE`. Model weights are not
distributed; each model is downloaded under its own terms. Pointers:
[LocalMind](https://github.com/NakliTechie/LocalMind) · `CHANGELOG.md`.
