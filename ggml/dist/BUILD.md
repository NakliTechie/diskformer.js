# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `e13de0a3c` (upstream master of 2026-10-22 merged in; head size 512 flash attention on the GPU; a thread-free log in wasm), Emscripten 6.0.13 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    a860e077df70b8799d894a3d22885c4c5f5d384fefcddcea8c86ed0ae50b4460  df-chat.wasm
    0f8a1a7e3eb75ef8f999d553d2fc1a124c4de1b6c6edb597643486752bcdce08  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
