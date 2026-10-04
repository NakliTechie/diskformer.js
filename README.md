<h1 align="center">diskformer.js</h1>

**Keep model weights on disk in the browser, and page into WebGPU only what each step needs.**
A disk tier for in-browser inference — not another engine.

Plain ES modules. Chromium with WebGPU. No server, no build step, no telemetry.

> **Status: pre-release.** The code runs today inside [LocalMind](https://github.com/NakliTechie/LocalMind).
> v0.1 extracts it here with no behaviour change. `0.0.x` on npm is a name placeholder.

## Install

| How | Command |
|---|---|
| npm | `npm install diskformer` *(from v0.1)* |
| CDN | `import { … } from 'https://cdn.jsdelivr.net/npm/diskformer/+esm'` *(from v0.1)* |

TODO(v0.1): the first call — copy a weights file into OPFS once, open a GPU row cache or expert pool on your
`GPUDevice`, and read what a step needs.

## Why

Your model does not fit on the GPU, or it fits only by crowding out everything else. Browser engines load every
weight onto the GPU at load time, including the ones a step never touches: Gemma 4's per-layer embedding table is
read one row per token, and a mixture-of-experts model uses under a tenth of its weights per token.

diskformer.js keeps those weights in OPFS — the browser's private file system, read through sync access handles in
workers — and pages into a GPU cache only what each step needs. What it does today, inside LocalMind:

- **Gemma 4 E2B:** the 1.2 GB per-layer embedding table on disk; Chrome's GPU process drops from 4.27 GB to 2.07 GB on
  the live site, with identical output.
- **Qwen3.6-35B-A3B (36.9 GB), experimental:** experts paged from OPFS on a 24 GB Mac; greedy output matches
  llama.cpp's Metal backend 8/8 on 4- and 16-layer cuts of the same GGUF.

## Context

Extracted from LocalMind's SSD-streaming work (2026-10-02 to 2026-10-04): `opfs-reader.js` (OPFS reader pool and
writer), `moe-expert-stream.js` (GPU slot pool with eviction, pins, prefetch and mapped staging-ring uploads) and
`ple-opfs.js` (row file and GPU row cache). v0.1 scope is those two layers — store and residency — moved here with no
behaviour change and verified by LocalMind's existing gates.

## License

MIT · LocalMind: [github.com/NakliTechie/LocalMind](https://github.com/NakliTechie/LocalMind)
