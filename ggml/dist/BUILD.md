# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-disk-tier` at `3119cf53d`, Emscripten 6.0.1 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`. Rebuild (copies here as a post-build step):

    cmake --build ggml/build-wasm --target df-chat

Update the fork commit above whenever the build changes.
