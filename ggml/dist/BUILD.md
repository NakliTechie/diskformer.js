# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `5ac033a52` (upstream master of 2026-10-22 merged in; head size 512 flash attention on the GPU; a thread-free log in wasm; DeepSeek-V4.1 from draft PR 30817 with WebGPU shaders for its ops), Emscripten 6.0.13 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    9c532bdddb66821d7be2dff2d2008f9a976ca7b497574fa2f1520b4f9ab9ff58  df-chat.wasm
    0f8a1a7e3eb75ef8f999d553d2fc1a124c4de1b6c6edb597643486752bcdce08  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
