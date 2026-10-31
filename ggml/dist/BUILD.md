# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `5ac033a52` (upstream master of 2026-10-22 merged in; head size 512 flash attention on the GPU; a thread-free log in wasm; DeepSeek-V4.1 from draft PR 30817 with WebGPU shaders for its ops), Emscripten 6.0.13 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    4c4c2ac6e8a6dec6f30687b5ccaf20b1392e552580d69ba8bd87366c0b8dc9f1  df-chat.wasm
    30f14aa3c1b561476f9ad80137807712f84a0ad7f9cd7c1f292a317871091016  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
