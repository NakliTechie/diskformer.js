# Changelog

## 0.6.6 (2026-10-24)
- The chat demo (`ggml/chat.html`) lists Ling 3.0 Flash 124B (2 parts, 77 GB, 6 GB budget) and downloads a model
  published in parts part by part, each under its own name; stop and resume work across parts, and Remove deletes every
  part. Checked in the browser: the download starts on part 1 of 2 from Hugging Face, a stop shows 0.34 GB of 77.0 GB
  stored with Resume, and Remove leaves no file in the page's OPFS.

## 0.6.5 (2026-10-24)
- The browser build loads a split GGUF (`name-00001-of-0000n.gguf`, as published): the worker opens every part from
  OPFS and mounts them together, `df_read_file_batch` and `df_upload_batch_js` read experts from the part that holds
  them, and `df_plan` sums the paged bytes over all parts. `gate.html` and `scripts/chrome-gate.mjs` take every part.
  Ling 3.0 Flash (124B-A5B, `bailingmoe3`, 2 parts, 77 GB) in Chrome at a 6 GB budget: copied into OPFS in 126 s,
  loaded in 60 s (22 slots), 4.1-4.9 tok/s; 6/9 replies match llama.cpp on the CPU (4 identical, 2 where the CPU reply
  is a prefix of Chrome's), the other 3 are coherent and differ where two words score almost the same. Paging exact
  natively, 9/9 at 8 vs 32 slots; Granite 4.0 H Tiny rechecked 9/9 in Chrome.
- Native `df-chat <gguf> <slots> --record <refs.json> <out.json>` renders each gate conversation with the model's own
  template and records 64-token replies, for models whose reference cannot come from llama-server on Metal.
- `scripts/record-refs.mjs`: CPU references (`--ngl 0`) keep the weights memory-mapped and unrepacked
  (`-lm mmap --no-repack`), so a model larger than RAM loads; a server killed by a signal now stops the run instead of
  leaving it waiting for 30 minutes.

## 0.6.4 (2026-10-23)
- `engines/lfm2_5.js` (LFM2.5 230M/1.2B, `Lfm2Mobile`), `engines/gemma-4-e2b.js` (Gemma 4 E2B/E4B, `Gemma4Mobile`) and
  `engines/ternary_bonsai_2_27b.js` (`TernaryBonsai2`), from LocalMind main a02f06e. Held back in 0.6.0 because their
  webml-community Spaces showed no license; Chirag wrote them and released them under MIT (2026-10-23). All three
  import in Node; their GPU checks are LocalMind's (`scripts/bench-engines.mjs`), not rerun here. With these,
  diskformer holds every LocalMind engine.

## 0.6.3 (2026-10-23)
- `LingMoeSsd.load` reads the two parts' headers (20 MB from Hugging Face) only when an ingest follows; a complete
  store loads without them (found by LocalMind's live check through anyformer). Verified in a real browser by LocalMind
  (after a 77 GB ingest through anyformer v0.2.5, a reload made no requests to the part URLs; answers correct before and
  after). `qwen3_moe_ssd.js` exports `STORE_FORMAT` and `STORE_ROOT`.
- `scripts/sync-localmind.mjs` runs diskformer → LocalMind for every file now, engines included (imports rewritten
  from `../src/` to `./`); `--ref` is gone, since LocalMind no longer writes engines.
- Native driver reads a split GGUF (`name-0000k-of-0000n.gguf`, as published) part by part, so it is never merged.
  Ling 3.0 Flash (124B-A5B, 2 parts, 77 GB) on the llama.cpp path natively: replies identical at 8 and 32 slots on 3
  prompts, 32 tokens. The browser build refuses a split GGUF with a message for now. Rebuilt browser build rechecked:
  Granite 4.0 H Tiny 9/9 in Chrome.

## 0.6.2 (2026-10-23)
- `engines/ling_moe_ssd.js`: the Ling 3.0 Flash engine (inclusionAI, 124B-A5B, `bailingmoe3`: KDA linear attention +
  MLA, 512 experts), handed over by LocalMind (d210986); the first engine diskformer adopts rather than mirrors.
  `splitFetch` ingests the two-part GGUF as one file. `engines/iq_quants.js` gains Q6_K and Q5_0. LocalMind's gate:
  6-layer cut 8/8 vs llama.cpp Metal; full model 5/8 vs llama.cpp CPU over 32 tokens, layer 0 within 5 decimals of a
  float64 reference (CPU is 1-3% off); decode 5.53 tok/s, GPU 2.51 GB dense + 4.29 GB pool (M4 Pro). Here: imports in
  Node, LocalMind's no-GPU test (`test-ling.mjs`) passes on this copy; the GPU gate was not rerun here.

## 0.6.1 (2026-10-22)
- `Llama.chat(messages, { thinking: false })` renders the chat template with reasoning off (llama.cpp's
  `enable_thinking`), for models that reason first: Qwen3.6 35B-A3B natively answers directly instead of opening with a
  thinking process. Default unchanged (the template's own choice). Asked for by anyformer, whose other engines pass
  `enableThinking: false`. Native driver: `df-chat model slots prompt n --no-think`. In Chrome (through anyformer's
  diskformer-llama): Gemma 4 26B's explain-tcp chat with `thinking: false` is identical to `gemma.json`'s reply.

## 0.6.0 (2026-10-22)
- `engines/` is the one home of LocalMind's hand-written engines (Chirag 2026-10-22: diskformer holds every custom
  engine; anyformer routes; LocalMind loads through anyformer). New: `lfm2_moe_ssd.js`, `hy_v3_moe_ssd.js`,
  `flint_moe_ssd.js`, `iq_quants.js`, `qwen35_dense.js`, `ple-opfs.js`, `flashmap_drafter.js`,
  `ternary_bonsai_2_dflash.js`, `ternary_bonsai_2_mtp.js`, copied from LocalMind main b5c289d with imports rewritten;
  each was gated in LocalMind and imports cleanly here (Node). Not copied: `lfm2_5.js`, `gemma-4-e2b.js` and
  `ternary_bonsai_2_27b.js`, vendored from webml-community Spaces that publish no license.
- `scripts/sync-localmind.mjs --ref <commit>` reads LocalMind's engines from a commit instead of its working tree.
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
