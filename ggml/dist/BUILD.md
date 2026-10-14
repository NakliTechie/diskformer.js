# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `377491df7` (upstream master of 2026-10-14 merged in), Emscripten 6.0.1 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    701e17b9dca2fa6c86853daf724c08a217f16ad91ce10b3bce711ccf63f9172c  df-chat.wasm
    2267afb7b9ee76149748956cc2418202526e4f2144784268cf7bdabf5127e062  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
