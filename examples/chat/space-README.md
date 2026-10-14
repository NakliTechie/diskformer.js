---
title: diskformer chat
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: MoE models larger than your GPU memory, in your browser tab
---

# diskformer chat

Mixture-of-experts models larger than your GPU memory, running in your browser tab: llama.cpp's WebGPU backend with
the experts paged in from the browser's private storage on your disk (OPFS). The model downloads once; the GPU holds
the dense layers and a small cache of experts, and every other expert is read from the file as each token needs it.
In the list: Qwen3.6 35B-A3B, GLM-4.7-Flash, LFM2.5 24B-A2B, ERNIE 4.5 21B-A3B, gpt-oss 20B, Qwen3 30B-A3B, Mellum2,
Ling mini 2.0, LFM2 8B-A1B, Granite 4.0 H Tiny, OLMoE, and Qwen3-Next 80B-A3B (48 GB, a proof of concept), or a GGUF from
your own disk. 16 MoE architectures pass the checks; the table is in
[docs/coverage.md](https://github.com/NakliTechie/diskformer.js/blob/main/docs/coverage.md). The earlier hand-written engines (Gemma 4 26B-A4B, Qwen3.6 35B-A3B) are under `classic/`.

Needs Chrome or Edge 137 or later with WebGPU, and free disk space for the model. Open the app in its own tab, so its
storage is first-party: https://naklitechie-diskformer-chat.static.hf.space/

Source, measurements and gates: [github.com/NakliTechie/diskformer.js](https://github.com/NakliTechie/diskformer.js).
Built by `node scripts/build-space.mjs` from the repo. Each model downloads from Hugging Face under its own license.
