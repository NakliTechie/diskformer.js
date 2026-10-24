# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `ecaa47b29` (upstream master of 2026-10-22 merged in, which adds Flint), Emscripten 6.0.13 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    7321264cf3a553377501cf696443914a6e7aa2f940f654b0a8c103f82a68e647  df-chat.wasm
    aed3b46f273dfd25f2ccb698178b94c40fdfc3483d43d1187032b2a48594f0fd  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
