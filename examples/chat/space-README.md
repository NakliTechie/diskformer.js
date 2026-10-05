---
title: diskformer chat
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: A 14.4 GB model in 2.5 GB of GPU memory, in your tab
---

# diskformer chat

Gemma 4 26B-A4B (14.4 GB) or Qwen3.6 35B-A3B (36.9 GB) running in your browser tab with a GPU memory budget you
choose. The model downloads once into your browser's private storage (OPFS); the GPU holds the dense layers and a
small cache of experts, and the rest are read from disk as each token needs them.

Needs Chrome or Edge with WebGPU and shader-f16, and free disk space for the model. Open the app in its own tab,
so its storage is first-party: https://naklitechie-diskformer-chat.static.hf.space/

Source, measurements and tests: [github.com/NakliTechie/diskformer.js](https://github.com/NakliTechie/diskformer.js).
Built by `node scripts/build-space.mjs` from the repo. Model terms: Gemma ([ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)); Qwen3.6, Apache-2.0. Engine
kernels are ported from gemma4-webgpu (Apache-2.0, see `engines/NOTICE`).
