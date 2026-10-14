# ggml/dist

`df-chat.mjs` + `df-chat.wasm`: `ggml/driver.cpp` linked with llama.cpp + WebGPU, built for a browser worker. Committed
because diskformer.naklitechie.com and the HF Space serve them and their CI has no Emscripten; the npm package ships them
as `diskformer/llama`.

Built from the fork `NakliTechie/llama.cpp` branch `webgpu-diskformer` at `f97fbe00d`, Emscripten 6.0.1 (Homebrew),
emdawnwebgpu from the fork's `emdawnwebgpu_pkg`, all C++ with `-fwasm-exceptions`. Rebuild (copies here as a
post-build step):

    cmake --build ggml/build-wasm --target df-chat

sha256 (check a copy with `shasum -a 256`):

    7b3a6c7eeb9d4f4206a280a7748300219e816636ad09de0dbb9ef0092d6961d0  df-chat.wasm
    2267afb7b9ee76149748956cc2418202526e4f2144784268cf7bdabf5127e062  df-chat.mjs

Update the fork commit and the checksums whenever the build changes.
