# Changelog

## Unreleased
- Run-time LoRA in every native MoE engine. The kernels and loader move to `engines/lora.js`; `Qwen3MoeSsd` (so every
  engine) has `loadAdapter(source, { scale })` and `setAdapter(adapter | null)`. An engine registers each matmul an
  adapter may follow with `loraIo(tensor, layer, site, x, y)` (and `loraIoB` for its chunked prefill), and adds an
  `encodeLora` / `encodeLoraB` hook after it. Hooked: Gemma 4 (attention q/k/v/output, dense FFN gate/up/down),
  Qwen3.6 (DeltaNet `attn_qkv` / `attn_gate` / `ssm_beta` / `ssm_alpha` / `ssm_out`, attention q/k/v/output, shared
  expert gate/up/down), LFM2.5 (unchanged behaviour, now on the shared code), and the Qwen3 base engine (attention
  q/k/v/output; not gated, no Q8_0 Qwen3 file on this machine). An adapter on a matrix with no hook, or with no
  chunked-prefill hook in an engine that batches its prefill, is refused.
- `scripts/make-test-lora.py`: a seeded random LoRA GGUF in llama.cpp's format, for models with no published adapter;
  the gate then compares the engine with `llama-server --lora` on the same file. `scripts/record-refs.mjs --url`
  takes `--backend` to describe the running server.
- Engines carry `addBos` (from the GGUF's `tokenizer.ggml.add_bos_token`, as llama.cpp tokenizes a raw prompt);
  `test/browser/text-gate.html` uses it (Qwen3.6 sets it false) and runs `--engine gemma | qwen35 | lfm2`.
- Gates, one loaded engine each in Chrome, greedy, 64 tokens, 9 prompts: plain / adapter on / plain again.
  Gemma 4 26B-A4B Q4_0 with a random rank-32 adapter on 205 matrices, against Metal llama.cpp: 9/9, 8/9, 9/9 (the
  miss is at char 193 of 214, where the adapter first changes that reply; ours kept the plain branch). Qwen3.6
  35B-A3B Q8_0 with a random adapter on 310 matrices, against CPU llama.cpp (36.9 GB, mmap): 6/9, 9/9, 6/9 (the plain
  misses are the WebGPU engine against the CPU reference, with no adapter). LFM2.5 with the Pi adapter: 8/9, 9/9,
  8/9, as in 0.6.19. In every engine the second plain phase gives the same text as the first.

## 0.6.19 (2026-10-31)
- Run-time LoRA in the native LFM2.5 engine: `await engine.loadAdapter(url | Blob | bytes, { scale })` reads a LoRA
  GGUF (llama.cpp's format: `W.lora_a` / `W.lora_b`, F16) onto the GPU, and `await engine.setAdapter(adapter | null)`
  switches it on or off between turns, with no reload and one base download for every adapter. Each adapted matmul
  is followed by y += scale · B(A·x) (new kernels `loraA` / `loraB`, batched `loraAT` / `loraBT`), scale = alpha / rank
  as llama.cpp's `--lora`. Adapters on the routed experts are refused. harness-hub's Pi adapter: 106 matrices,
  37.4 MB on the GPU, loaded in 0.2 s.
- Gate (`test/browser/text-gate.html --query adapter=…&ref2=…`, one engine, Chrome, 4 GiB pool, greedy, 64 tokens):
  plain 8/9 against llama.cpp; adapter on 9/9 against `llama-server --lora` (the merged Q8_0-dense file gave 8/9);
  adapter off again 8/9, the same text as before it was switched on. Decode speed was the same in all three phases
  in this run (8.2, 8.4, 8.3 tok/s median, with full logits read back each token; not a benchmark).
- Demo page `examples/lora/` (site: `/lora/`): LFM2.5-24B-A2B with harness-hub's Pi adapter, a Plain / Pi switch and
  "Ask both, side by side". Headless check (`?auto=1`): switch 0.2 ms; plain, Pi, plain again give 256 tokens each at
  11.2, 10.9 and 11.5 tok/s; the second plain reply is identical to the first. `scripts/build-space.mjs` adds it.
- `engines/ternary_bonsai_2_mtp.js`: per-model MTP heads (Bonsai 2 27B and 4B); the head kernels take the query-head
  count (commit caf691c, from another session; its own checks are not part of this release's gates).

## 0.6.18 (2026-10-31)
- Reasoning off now holds for templates that always open a thought block. LFM2.5-24B-A2B's template ends the
  generation prompt with `<think>` and ignores `enable_thinking`, so `llama.chat(messages, { thinking: false })` still
  returned reasoning and a `</think>` tag. With reasoning off, the driver now closes that block empty (`\n</think>\n\n`,
  the form Qwen3's and Gemma 4's templates render) and bans the `<think>` token for the reply (logit bias -inf),
  because with the empty block alone the model opened a new one in 1 of 9 gate conversations. Reasoning on is
  unchanged: the same prompt, and no bias.
- `scripts/chrome-gate.mjs --no-think`: each conversation through the chat template with reasoning off; a reply passes
  with text and no `<think>` tags. LFM2.5-24B-A2B in Chrome (4 GB budget): 9/9 with the Pi adapter, 9/9 without;
  reasoning on with the adapter 8/9 against llama.cpp, as in 0.6.17.

## 0.6.17 (2026-10-31)
- LoRA adapters on the llama.cpp path: `Llama.load({ name, lora, loraScale })` applies a LoRA GGUF stored in OPFS
  beside the model (`df_set_lora` in `ggml/driver.cpp`, llama.cpp's `llama_adapter_lora_init`, as `llama-server
  --lora` does); `df_plan` counts the adapter's bytes. An adapter on the dense layers leaves expert paging alone.
  `scripts/chrome-gate.mjs --lora <adapter.gguf> [--lora-scale s]`; natively `DF_LORA=<adapter.gguf>`. ggml/dist rebuilt.
- `engines/lfm2_moe_ssd.js` runs a LoRA-merged LFM2.5-24B-A2B: when the file stores the dense matrices as Q8_0 it uses
  the Q8_0 matmul kernels for them, while the experts stay on the Q4_0 kernels. The merged file is made with llama.cpp:
  `llama-export-lora` (touched matrices F16), then `llama-quantize --allow-requantize --token-embedding-type q6_k
  --tensor-type _exps=q4_0 <merged> <out> Q8_0`, which copies every tensor already at its target type byte for byte.
  For harness-hub's Pi adapter (`harness-hub/LFM2.5-24B-A2B-pi-GGUF` at a994cf7, rank 32, alpha 64) on the QAD-Q4_0
  base: 13.95 GB, the 106 touched matrices Q8_0, the other 322 tensors (12.91 GB of experts among them) identical to
  the base.
- Gates (M4 Pro, Chrome, 4 GB budget or pool, greedy, 64 tokens, the 9 chat conversations), against llama.cpp b11669
  on Metal (`examples/chat/refs/lfm2.5-24b-a2b-pi-base.json`, `-pi-lora.json`; the adapter changes 3 of 9 replies):
  llama.cpp path in Chrome, base + adapter: 8/9 identical (without the adapter 7/9, the path's own baseline); native
  engine on the merged file: 8/9 (on the base file 8/9). Every miss is a late split (character 112 or later) where
  Metal llama.cpp on the merged file also splits from the run-time adapter (`-pi-merged.json`: 7/9).
- `scripts/record-refs.mjs`: `--lora`, and `--url <server> --lora-request id:scale` to record from a running
  llama-server. `test/browser/text-gate.html`: a native engine's greedy text against a refs file (adds BOS as
  llama-server's /completion does).

## 0.6.16 (2026-10-31)
- Chrome 156 fix: Ternary Bonsai 2 27B, Gemma 4 E2B, LFM2.5 and the IQ-quant dense path failed to load on Apple GPUs
  with "Mixed precision subgroup matrix multiply is not supported by the device". Their prefill kernels multiply f16
  subgroup matrices into an f32 result; Apple reports only f16-f16 and f32-f32 configurations, and Dawn stopped
  compiling the mixed pair in Chrome 156. New `engines/subgroup-matrix-guard.js` compiles one tiny mixed kernel on
  the first adapter request and, when that fails, hides `chromium-experimental-subgroup-matrix` from the adapter, so
  those engines take their existing non-matrix prefill path (slower prefill, same output path as GPUs without
  subgroup matrices). Imported by `ternary_bonsai_2_27b.js`, `gemma-4-e2b.js`, `lfm2_5.js` and `qwen3_moe_ssd.js`.
  No other engine uses subgroup matrices. Synced to LocalMind with the other engine files.

## 0.6.15 (2026-10-30)
- Clover-35B-A3B-Preview (Diffuspace, Apache-2.0), a block-diffusion model on Qwen3.6-35B-A3B, from the SSD:
  `engines/clover_moe_ssd.js` (`CloverMoeSsd`, GGUF arch `clover_moe`; pinned to the authors' Q4_0 file, 19.6 GB,
  `Diffuspace/Clover-35B-A3B-Preview-GGUF` at 32e22a0). The commit pass is Qwen35MoeSsd's chunked prefill. New is the
  preview (denoising) pass over a 32-token block: the DeltaNet conv also looks right inside the block, the delta rule
  becomes a softmax over the block's keys plus a "history" slot read from the clean recurrent state, full attention
  sees the whole block both ways, and one coreset of experts (90% of the block's routing mass) serves the block. A
  preview writes nothing to the committed cache.
- `forward({ mode: 'preview' | 'commit', start, ids, wantLogits })` is the callback kohra.js's denoise loop drives;
  previews return all logits ('all', T x vocab, row t scores slot t) or the GPU top-1 and its confidence ('top1').
  `generate()` runs the loop with Clover's reference settings (threshold 0.95, its generation config's penalties).
  kohra's loop is vendored unchanged as `engines/vendor/kohra-denoise.js`; `node scripts/sync-kohra.mjs` checks it.
- Clover's GGUFs store the DeltaNet decay as `A_log` (Qwen3.6's store -exp(A_log)); the engine converts it at load.
- `qwen35_moe_ssd.js` and `qwen3_moe_ssd.js` accept the `clover_moe` architecture (`Qwen35MoeSsd.archs`).
- Gates: `scripts/clover-oracle.py` runs the authors' modeling_clover_moe.py on CPU (float32, weights dequantized from
  a 4-layer cut made by `scripts/gguf-cut.py`); `test/browser/clover-states.html` replays prompt, preview, commit,
  preview on the engine; `scripts/clover-check.py` compares. Measured: DeltaNet layers 1-2e-6 relative error,
  the attention layer 2e-5 (f16 KV), logits 1.7e-4, top-1 identical on 64 of 64 rows ('all' and 'top1'). Full model
  in Chrome (`test/browser/clover-chat.html`, 1 GB expert pool): "What is the capital of France? Answer in one
  sentence." gives "The capital of France is Paris." (2 previews); a Fibonacci request gives a coherent start of an
  iterative implementation (64 tokens, 38 previews). Not benchmarked yet.

## 0.6.14 (2026-10-30)
- LFM3-Omni speech out is 3x faster in a background tab. The inverse STFT ran a naive inverse DFT on the main thread
  (15-18 ms of CPU per 80 ms speech frame; 75 ms when the page runs in the background, as a hidden Claude desktop
  browser pane does), so SansadSaar's voice chat got 0.54x real time. It now runs a mixed-radix FFT (1280 = 2^8 * 5):
  1.2 ms per frame, output bit-identical to the old one (138,240 of 138,240 samples on 432 random rows). Measured on
  the M4 Pro, one sampled 13.5 s answer: 1.50-1.61x real time in a hidden browser pane (was 0.54x), 1.71-1.77x
  visible, 1.41-1.45x in headless Chrome under other CPU load (was 1.30-1.38x). Per frame now: decoder step about
  20 ms, depth decoder (3 MaskGIT passes) about 20 ms, codec 1 ms, inverse STFT 1.2 ms. No new options; codes and
  text are unchanged (speech gate: first frame difference at frame 7, as in 0.6.13).
- `test/browser/lfm3-omni-speed.html`: times each speech stage per frame and reports the real-time factor.

## 0.6.13 (2026-10-30)
- LFM3-Omni: one reply at a time. `generate()` and `greedyIds()` share the decoder's caches and onnxruntime's sessions;
  two replies overlapping on WebGPU failed ("Failed to download data from buffer") or hung (the NakliOS assistant: a
  text reply with tools left running, then a speech reply). A call made while another runs now waits its turn.
- Gate `test/browser/lfm3-omni-overlap.html`: text then speech on one engine, and speech started while text runs; both
  stream audio, nothing throws or times out.

## 0.6.12 (2026-10-30)
- LFM3-Omni-2B speaks: `generate(messages, { output_modalities: ['text', 'audio'], onAudio })` runs Liquid's
  interleaved mode (6 text tokens, 12 speech frames in turn), the depth decoder's MaskGIT (12 codes a frame) and the
  codec decoder, then an inverse STFT to 24 kHz Float32 PCM, streamed about once a second. Options `audio_temperature`
  (0.4), `audio_top_k`, `audio_repetition_penalty` (1.2), `audio_repetition_window` (64), `audio_chunk_frames`, `seed`;
  `load({ speech: true })` fetches the two speech graphs (depth_decoder 336 MB, codec_decoder 179 MB, fp32) up front,
  else the first speech call does. Text-only output is unchanged and stays the default.
- Gate `test/browser/lfm3-omni-speech.html`: text identical to Liquid's runner slot for slot; speech codes exact for
  the first frames, then fp16 near-ties on WebGPU flip a code and later frames differ (M4 Pro: frames 0-6 exact), so
  the gate asks for 4 exact frames, the same length and RMS within 10%; a sampled 13.5 s answer took 9.6 s, first
  audio after 2.3 s (speech prompt included). The inverse STFT matches numpy to 5e-10 of peak. `--wav` writes the
  replies for listening.

## 0.6.11 (2026-10-30)
- Republishes 0.6.10's tagged files. npm 0.6.10 was packed from a working tree that held uncommitted, unchecked
  speech-output work in `engines/lfm3_omni.js` (854 lines against the tag's 628); 0.6.11 carries the tag's engine.
  No code change from v0.6.10.

## 0.6.10 (2026-10-30)
- LFM3-Omni-2B (`engines/lfm3_omni.js`, `Lfm3Omni`): LiquidAI's ONNX export (decoder_q4f16, vision_encoder_q8,
  audio_encoder; 2.14 GB) on onnxruntime-web 1.30.0's WebGPU EP, because transformers.js 4.3.2 has no `lfm3_omni`
  class. Text, image and speech in (16 kHz, at most 30 s a clip), text out; streaming (`onText`, `onToken`, a
  transformers.js streamer), an AbortSignal, `enable_thinking`, `tools` (tool calls stay raw text). Model files go to
  Cache Storage 'diskformer-models'; `Lfm3Omni.cached()` / `deleteCache()` list and remove them by name.
- Gate `test/browser/lfm3-omni.html`: prompt ids and 32 greedy tokens equal Liquid's runner on onnxruntime's CPU EP
  (`scripts/lfm3-omni-ref.py`) for text, image and speech, 3/3, plus a reload from Cache Storage with no network
  requests. onnxruntime-web 1.31.1 diverged at token 28 on one case, so the engine pins 1.30.0.
- anyformer 0.4.13 runs it as `MODELS.chat.LFM3_OMNI_2B` (back end `diskformer-onnx`).
- `examples/run-demo.mjs --query` passes extra page parameters.

## 0.6.9 (2026-10-24)
- DeepSeek V4.1 Flash 1017 (763B, 6 of 384 experts active) in the chat demo as a tech demo: ISTA-DASLab's GSQ-RCO
  IQ1_M (4 parts, 120.7 GB) plus its separate Engram tables (110.6 GB), 231.3 GB in all, at an 8 GB GPU budget.
  In Chrome it loaded in 173 s and answered one prompt as the native build does ("The capital of France is
  **Paris**."); about 0.4 tok/s natively from an external SSD (1.3 GB of experts read per token). Layers 0-2 on
  llama.cpp's CPU backend match a float64 reference within llama.cpp's own rounding. Not gated on 9 prompts.
- The fork (`webgpu-diskformer` at `5ac033a52`) takes draft PR 30817 (DeepSeek-V4.1) with a loader for ISTA's
  layout (separate Engram file, unfused gate, u32 candidate layer) and WebGPU shaders for its ops:
  LIGHTNING_INDEXER, DSV4_HC_PRE/POST/COMB, TOP_K_MASK, POOL_1D, ROPE_BACK, f16 CONCAT and BF16 CPY.
  Regression gates on the new build: Granite 4.0 H Tiny 9/9 and Nemotron 3.5 Bolt E4B 9/9 in Chrome.
- `load({ extra })` mounts stored side files beside the GGUF; the chat demo downloads a catalog entry's side
  files with its parts. `scripts/chrome-gate.mjs --extra` and `--dist`.

## 0.6.8 (2026-10-24)
- Nemotron 3.5 Bolt E4B (dense, `gemma4`, 6.18 GB) in the chat demo: 9/9 replies identical to llama-server on WebGPU
  in Chrome, 30 tok/s at a 5 GB budget. Three fixes made it run in a browser:
  - Head size 512 attention (Gemma 4's global layers) stays on the GPU: the fork's WebGPU flash attention reported the
    op unsupported when the subgroup matrix path had no room, though its pipeline falls back to the tile path
    (fork `bb798fd31`).
  - The worker reads the mounted GGUF at most 1 GiB at a time, because `FileReaderSync` fails on a slice over 2 GiB
    (Bolt's per-layer embedding table is 2.8 GiB); before this, Bolt did not load in a browser at all.
  - llama.cpp's common log prints synchronously in wasm without pthreads (fork `e13de0a3c`); its worker thread threw
    on construction and stopped speculative decoding in the browser.
- Speculative decoding with a drafter GGUF (DFlash, DSpark, MTP: the type comes from the GGUF):
  `load({ draft })` in `diskformer/llama`, `df_set_draft`, and `DF_DRAFT` natively; stats add `drafted`, `accepted`,
  `draft_ms`, `verify_ms`. Default draft length 3 (`DF_DRAFT_N_MAX`), greedy-identical on WebGPU. Bolt's DFlash
  drafter gives 9/9 identical replies and 65% acceptance, but 24.7 tok/s against 30.0 plain, because a 4-token verify
  on WebGPU costs about 3 decodes; the chat demo therefore loads Bolt without it.
- `df_plan` plans a dense model (no experts) and counts a drafter's bytes.
- `scripts/chrome-gate.mjs --draft <gguf>` and `--log <file>` (Chrome's log, with the driver's lines and C++
  exception messages, which the worker now decodes); `scripts/record-refs.mjs --draft --spec-type --draft-max`.

## 0.6.7 (2026-10-24)
- Ingest on a slow or flaky link (`src/ingest.js`, every hand engine's download): the GGUF header streams with
  `ingest-header` progress instead of arriving silently in one 16 MB piece; a range that fails mid-stream (connection
  dropped, HTTP 5xx or 429) is retried from the last byte consumed, up to 6 times with pauses of 1 to 15 s, and each
  retry reports `ingest-retry`; an abort, a 4xx and a write failure still end the ingest. `ingestProgress` maps both to
  `weights` events (`phase: 'header' | 'retry'`), so a host watchdog sees the download moving. Found by LocalMind's
  production run at 0.1 MB/s (Flint: 3 minutes without a progress event; LFM2.5 24B: "network error" after 0.68 GB).
  Tests: a dropped connection resumes to the same bytes as a clean ingest; a 404 is not retried.

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
