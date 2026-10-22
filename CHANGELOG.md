# Changelog

## Unreleased
- `DF_OUTPUT_CPU=1` (driver setting, `env` in `Llama.load`) keeps the token table on the CPU, so a model whose output
  layer reuses it computes its logits there. Gemma 4 26B-A4B: the fixed GPU bill drops from 2,106 to 1,358 MiB (token
  table 577.5 MiB, logits buffer ~171 MiB), but decode in Chrome falls from 9.3 to 5.4 tok/s at 8 slots, because the
  browser build's CPU work runs on one thread; replies unchanged (8/9 both ways). Off by default.
- `scripts/chrome-gate.mjs --env K=V,…` passes driver settings to the gate page.

## 0.5.4 (2026-10-22)
- `diskformer/llama`: the GPU budget plan picks the physical batch size (llama.cpp's `n_ubatch`) as the largest of
  512, 256, 128 and 64 that leaves room for one token's experts. The compute buffers scale with it (Gemma 4 26B-A4B:
  542 MiB at 512, 271 at 256, 135 at 128), so a tight budget now runs instead of failing: Gemma 4 runs at 3 GB
  (8 slots, batch 256; 8/9 replies identical to llama.cpp in Chrome and natively), where it needed about 3.4 GB. A
  smaller batch slows long prompts (649 tokens: 95 tok/s at 512, 46 at 128), so ample budgets keep 512.
  `DF_UBATCH` fixes the size. `scripts/chrome-gate.mjs --budget GB`.
- Flint 40B-A2.5B (RecursiveSI, architecture `flint`, Q4_1, 26.0 GB) runs on the llama.cpp path and is in the demo
  list. Paging is exact (9/9 replies identical at 8 and 32 slots natively); in Chrome at a 4 GB budget (26 slots),
  6/9 replies are identical to llama.cpp on the CPU, and the other 3 part at word choices inside the reasoning,
  1.8-3.4 tok/s. 17 architectures now pass both checks.
- Browser build (`ggml/dist`) rebuilt on the fork after merging 178 upstream commits (`ecaa47b29`, adds Flint),
  Emscripten 6.0.13.
- `scripts/sync-llama-fork.sh` puts Homebrew on PATH, so the weekly launchd run finds cmake (the 2026-10-18 run failed
  without it).
- Engines: `gemma4_moe_ssd.js` synced from LocalMind (embedding-kernel hooks); chat gate 9/9.

## 0.5.3 (2026-10-15)
- Engines synced from LocalMind: the Q4_0 and Q6_K kernels are shared from `engines/qwen3_moe_ssd.js`;
  `engines/qwen35_moe_ssd.js` reads Q4_0 files and carries MiniCPM-V 4.7's image encoder
  (`engines/minicpmv_vision.js`, new). Gemma 4 26B-A4B chat gate: 9/9 replies identical to llama.cpp at a 2.5 GB
  budget (2.02 GB on the GPU).
- docs/coverage.md: Velvet-Eclipse 4x12B (Mixtral-style `llama` MoE) runs with paged experts, too slowly to be
  practical (0.74 tok/s, 1.4 GB read per token).

## 0.5.2 (2026-10-14)
- `diskformer/llama` works when imported from a CDN: a `Llama` on another origin starts its worker from a same-origin
  stub that imports `worker.mjs`, because browsers refuse a cross-origin worker URL. Checked from a localhost page
  importing only from jsDelivr: download, load and a chat reply from Granite 4.0 H Tiny.
- README: a quick start that runs as pasted; the llama.cpp path leads.
- docs/coverage.md: diffusion MoE models are not supported. Paging is exact for them, but each denoising step reads
  nearly all experts (about 3.9 GB per step for LLaDA-MoE 7B-A1B); llama.cpp's WebGPU output for it degrades (open).

## 0.5.1 (2026-10-14)
- The browser build (`diskformer/llama`) is rebuilt on the fork after merging upstream llama.cpp (fork `377491df7`, 167
  upstream commits including the MUL_MAT_ID rewrite #30451); checksums in `ggml/dist/BUILD.md`. Every gated model was
  rechecked in Chrome on it: 15 of 15 at their earlier result or within one reply (gpt-oss 8/9 to 7/9).
- Chat prompts are tokenized part by part: template text with its special tokens, user text as plain text, so text a
  user types cannot become a control token.
- Coverage: 16 architectures pass both checks, adding Nemotron 3 Nano (nemotron_h_moe) and Ling 3.0 tiny (bailingmoe3),
  and Gemma 4 on the llama.cpp path (8/9). Table: `docs/coverage.md`.
- Demo: Qwen3.6 35B-A3B, GLM-4.7-Flash, LFM2.5 24B-A2B, ERNIE 4.5 21B-A3B, Ling mini 2.0 and Mellum2 in the model list.

## 0.5.0 (2026-10-14)
- Added: `diskformer/llama`, the llama.cpp path in the package. `ggml/llama.js` (`Llama`: download, list, remove, load,
  chat, complete, abort) drives `ggml/worker.mjs`, which runs `ggml/dist/df-chat.wasm`: llama.cpp with WebGPU, the MoE
  experts paged in from OPFS. Built from the fork NakliTechie/llama.cpp, branch `webgpu-diskformer` (renamed from
  `webgpu-disk-tier`); commit and sha256 in `ggml/dist/BUILD.md`.
- Width: gated on this path (paged replies identical across slot counts natively, then Chrome against llama.cpp):
  gemma4, qwen3moe, qwen3next, qwen35moe, qwen3vlmoe, olmoe, lfm2moe (LFM2, LFM2.5), granitehybrid, gpt-oss (MXFP4),
  ernie4_5-moe, deepseek2 (GLM-4.7-Flash), mellum, bailingmoe2, cohere2moe. References in `examples/chat/refs/`.
- Fixed: chat templates that use `{% break %}` killed the worker (all C++ now built with wasm exceptions); a C++
  exception is reported with its message; a template that fails no longer stops the model loading.
- Fixed: a download resumed after a reload before its first chunk finished kept half-written chunks (a chunk counts as
  finished only after its bytes are flushed).
- Demo: reasoning (gpt-oss channels, `<think>` blocks) shows as a collapsible Thought; a stored file's name renders as
  text; the classic page takes `?url=` only on localhost.

## 0.4.1 (2026-10-13)
- Docs only: the README now leads with the llama.cpp path (Qwen3-Next 80B-A3B, a 48 GB file, in under 5 GB of GPU
  memory; OLMoE and LFM2 in real time) and the recording. No code changes.

## 0.4.0 (2026-10-13)
- Changed: errors are now `DiskformerError`s with a `code` (see below). A caller that checked `e.name === 'AbortError'`
  after an aborted ingest should check `isDiskformerError(e, 'LOAD_ABORTED')`; messages are unchanged.
- Coded errors (`src/errors.js`): `DiskformerError` with a `code`, checked by name and code through
  `isDiskformerError(e, code)`. `ingestGguf` throws `LOAD_ABORTED` when its signal aborted, else `LOAD_FAILED` with the
  original as `cause`.
- `readStore(key)` and `listStores()`: a store is complete only when every file it names is on disk at full size;
  zero-byte or short files left by a crash make it `invalid`. `examples/chat` lists the stored models and removes one
  at a time.
- `engines/`: `UNSUPPORTED_DEVICE`, `BUDGET_TOO_SMALL` (with the bytes in `detail`), `CONTEXT_FULL`,
  `EXECUTION_FAILED`, `RESOURCE_DISPOSED`. `generate()` and `greedy()` on one engine run one at a time (they share the
  KV cache); `dispose()` waits for the step in flight. A load that fails destroys its GPU device. An invalid store is
  ingested again instead of loaded.
- Gates that run from this repo: `test/browser/greedy.html` (token by token vs llama.cpp, references in `test/refs/`),
  `test/browser/states.html` + `test/state-check.py` (per-layer states vs LocalMind's numpy pass).
- New helpers: `opfsFileSize(path)`, `opfsSubdirs(path)`.
- Not in the npm package, but new in the repo: the demo at diskformer.naklitechie.com now runs llama.cpp's WebGPU
  backend with experts paged from OPFS (`ggml/`; LFM2 8B-A1B, OLMoE 1B-7B, Qwen3 30B-A3B, Qwen3-Next 80B-A3B); the
  hand-engine page moved to `/classic/`.

## 0.3.0 (2026-10-07)
- `fileFetch(blob)`: ingest a GGUF that is already on disk (a `File` the user picked, e.g. from a Hugging Face or
  LM Studio cache) through `ingestGguf({ fetch: fileFetch(file) })`, with no download. The store is byte-identical to a
  download's, and an interrupted file ingest resumes from the same file (`test/ingest-resume.test.mjs`).
- `engines/`: `load({ localFile })` ingests the pinned GGUF from a local file, refusing a file of another size.

## 0.2.0 (2026-10-05)
- `engines/`: reference WebGPU engines for Gemma 4 26B-A4B (`Gemma4MoeSsd`), Qwen3.6 35B-A3B (`Qwen35MoeSsd`) and
  Qwen3-30B-A3B (`Qwen3MoeSsd`), copied from LocalMind with import paths rewritten (`scripts/sync-localmind.mjs`).
  `gpuBudgetBytes` sizes the expert cache to fit a GPU memory budget; `root` picks the OPFS folder. Kernels ported
  from gemma4-webgpu (Apache-2.0, `engines/NOTICE`).
- `examples/chat`: one page that downloads a model into OPFS and chats under a GPU budget. Gemma 4 26B-A4B (14.4 GB)
  at a 2.5 GB budget: 2.02 GB on the GPU, 9/9 replies identical to llama.cpp b9830 Metal.
- `ingestGguf` resumes an interrupted ingest from its last 1 GiB checkpoint, only into the same source (header, size,
  ETag) and output layout, and closes every file when it fails. Progress events carry `resumedFrom`.
- `examples/run-demo.mjs`: a `chat` gate (`--budget`, `--models`, `--site` for a deployed copy), browser test pages,
  local GGUF serving with HTTP Range (416 / 400 / symlink-confined).
- Tests: `test/ingest-resume.test.mjs` (fake OPFS: abort twice, changed source, failures);
  `test/browser/ingest-resume.html` (real OPFS, every byte vs a clean ingest); `test/browser/server.html`.
- `scripts/build-space.mjs` and a Hugging Face Space: https://huggingface.co/spaces/naklitechie/diskformer-chat

## 0.1.0 (2026-10-05)
- Store: `OpfsReaderPool`, `OpfsWriter` and OPFS helpers (byte-identical to LocalMind's `opfs-reader.js`); `ingestGguf`
  with `root` and `format` options (extracted from LocalMind's `qwen3_moe_ssd.js`; output byte-identical on an 8 GB
  Gemma 4 GGUF); `parseGguf`, `tensorBytes`, `splitQ8`, `splitQ4`; `measure()` (new).
- Residency: `ExpertStreamer` / `RecordPool` (byte-identical to LocalMind's `moe-expert-stream.js`); `RowFile`,
  `RowCache`, `fingerprint` with a `root` option (from LocalMind's `ple-opfs.js`, Gemma helpers left in LocalMind).
- Tests: Node unit tests (`npm test`); a headless-Chrome demo that checks every resident row on the GPU.

## 0.0.1 (2026-10-04)
- Name placeholder on npm (README and LICENSE only).
