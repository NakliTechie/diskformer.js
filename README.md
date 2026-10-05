<h1 align="center">diskformer.js</h1>

**Keep model weights on disk in the browser, and page into WebGPU only what each step needs.**
A disk tier for in-browser inference — not another engine.

Plain ES modules, no dependencies, no build step. Chromium with WebGPU. Weights stay in the user's own browser storage.

[![npm](https://img.shields.io/npm/v/diskformer?style=flat-square&color=555)](https://www.npmjs.com/package/diskformer)
[![license](https://img.shields.io/badge/license-MIT-555?style=flat-square)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-555?style=flat-square)](package.json)

## Install

| How | Command |
|---|---|
| npm | `npm install diskformer` |
| CDN | `import { RowFile, RowCache } from 'https://cdn.jsdelivr.net/npm/diskformer@0.1/index.js'` |
| Copy | the `src/` files and `index.js`; nothing else is needed |

The first call keeps a table on disk and a few thousand of its rows on the GPU. Row files use OPFS sync access
handles, so this runs in a dedicated worker:

```js
import { RowFile, RowCache, fingerprint } from 'diskformer';

const file = await RowFile.open({ key: 'my-model', rowBytes, rows });     // OPFS: diskformer/my-model/rows.bin
const fp = fingerprint([weights]);                                          // changes when the weights change
if (!file.matches(fp)) await file.write(fp, (row0, n, dst) => dst.set(weights.subarray(row0 * rowBytes, (row0 + n) * rowBytes)));
await file.openRead();
const cache = new RowCache({ file, slots: 4096, planes: [{ offset: 0, bytes: rowBytes, buffer: gpuRows }], queue: device.queue });
const slots = cache.lookup(tokenIds);    // rows resident on the GPU; your kernel reads gpuRows at slot × rowBytes
```

Nothing to configure, no server. `examples/embedding` is the whole loop in one page, with the WGSL kernel that reads
the slots: a 128 MB table on disk, 1 MB of it on the GPU.

## Why

Your model does not fit on the GPU, or it fits only by crowding out everything else. Browser engines load every
weight onto the GPU at load time, including the ones a step never touches: Gemma 4's per-layer embedding table is
read one row per token, and a mixture-of-experts model uses under a tenth of its weights per token.

diskformer keeps those weights in OPFS, the browser's private file system, read through sync access handles in
workers. It pages into a GPU cache only what each step needs. It is the store and the residency layer; your engine
keeps its own kernels and decides what a step needs. Chromium only (OPFS sync handles and WebGPU).

## A table on disk

`RowFile` holds fixed-size rows in one OPFS file with a manifest, written once and validated by a fingerprint.
`RowCache` keeps `slots` rows on the GPU, split into planes when a row's parts live in different buffers, with an
O(1) LRU and an optional GPU id→slot map that kernels can read. `warm(row0, count)` preloads the rows you expect.

## Experts on disk

`ingestGguf({ url, key, plan })` streams a GGUF over HTTP ranges into OPFS in your engine's layout: `plan.layout`
says where each tensor goes, `plan.units` lists the byte ranges to copy or split (Q8_0 and Q4_0 blocks become a
value plane and an f16-scale plane, bit-exact). `ExpertStreamer` (also exported as `RecordPool`) keeps a fixed GPU
slot pool of expert records: `ensure(layer, ids)` returns pinned slots, `prefetch(layer, guesses)` never blocks, and
uploads go through a mapped staging ring.

## Measure the disk

`measure({ path, recordBytes })` reports sequential MB/s and random whole-record reads, one at a time and in bursts
of 8, through the reader pool. OPFS speed varies a lot between machines; measure where your users run.

## Commands

```bash
npm test                               # Node unit tests: record pool, row cache, GGUF parse and block splits, ingest helpers
node examples/run-demo.mjs rows        # headless Chrome: a 64 MB table on disk, 2,048 rows on the GPU, every byte checked
node examples/run-demo.mjs embedding   # the full loop: lookup → slots → a WGSL gather kernel, every output value checked
```

## Verify it yourself

`npm test` runs the unit tests with fake GPU queues. `node examples/run-demo.mjs` writes a table to OPFS, looks up
8,000 ids, reads the GPU back and fails on any byte or map entry that differs from the source.

The code runs in [LocalMind](https://github.com/NakliTechie/LocalMind) on its live site. `opfs-reader.js` and
`expert-stream.js` are byte-identical to LocalMind's copies. `rows.js`, `gguf.js` and `ingest.js` are extracted from
it. Ingesting an 8 GB Gemma 4 GGUF with this `ingestGguf` produced files byte-identical to LocalMind's (2026-10-05).

## License

MIT. Pointers: [LocalMind](https://github.com/NakliTechie/LocalMind) (where it runs) · `CHANGELOG.md`.
