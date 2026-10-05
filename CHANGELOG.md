# Changelog

## 0.2.0 (2026-10-05)
- `engines/`: reference WebGPU engines for Gemma 4 26B-A4B (`Gemma4MoeSsd`), Qwen3.6 35B-A3B (`Qwen35MoeSsd`) and
  Qwen3-30B-A3B (`Qwen3MoeSsd`), copied from LocalMind with import paths rewritten (`scripts/sync-localmind.mjs`).
  `gpuBudgetBytes` sizes the expert cache to fit a GPU memory budget; `root` picks the OPFS folder. Kernels ported
  from gemma4-webgpu (Apache-2.0, `engines/NOTICE`).
- `examples/chat`: one page that downloads a model into OPFS and chats under a GPU budget. Gemma 4 26B-A4B (14.4 GB)
  at a 2.5 GB budget: 2.02 GB on the GPU, 9/9 replies identical to llama.cpp b9830 Metal.
- `ingestGguf` resumes an interrupted ingest from its last 1 GiB checkpoint, and releases its files when it fails.
- `examples/run-demo.mjs`: a `chat` gate, browser test pages, and local GGUF serving with HTTP Range.
- `test/browser/ingest-resume.html`: abort at 60%, resume, compare every byte with a clean ingest.

## 0.1.0 (2026-10-05)
- Store: `OpfsReaderPool`, `OpfsWriter` and OPFS helpers (byte-identical to LocalMind's `opfs-reader.js`); `ingestGguf`
  with `root` and `format` options (extracted from LocalMind's `qwen3_moe_ssd.js`; output byte-identical on an 8 GB
  Gemma 4 GGUF); `parseGguf`, `tensorBytes`, `splitQ8`, `splitQ4`; `measure()` (new).
- Residency: `ExpertStreamer` / `RecordPool` (byte-identical to LocalMind's `moe-expert-stream.js`); `RowFile`,
  `RowCache`, `fingerprint` with a `root` option (from LocalMind's `ple-opfs.js`, Gemma helpers left in LocalMind).
- Tests: Node unit tests (`npm test`); a headless-Chrome demo that checks every resident row on the GPU.

## 0.0.1 (2026-10-04)
- Name placeholder on npm (README and LICENSE only).
