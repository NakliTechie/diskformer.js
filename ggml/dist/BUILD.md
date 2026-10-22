# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `ecaa47b29` (upstream master of 2026-10-22 merged in, which adds Flint), Emscripten 6.0.13 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    22d53dc1e8a6be1c723be179c044915d02639e3702dac513492a9bac5194f8c8  df-chat.wasm
    43dea00df2a5345efc65c6521b825032670447bac854d2d346aa82132b3b37f9  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
