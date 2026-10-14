# Model coverage on the llama.cpp path

Which mixture-of-experts (MoE) models run in a browser tab through `diskformer/llama` (llama.cpp's WebGPU backend from
the fork [NakliTechie/llama.cpp](https://github.com/NakliTechie/llama.cpp/tree/webgpu-diskformer), with the experts
paged in from OPFS), and how each was checked. Paging has no per-model code: it applies to every `*_exps` tensor, so
one passing model covers its architecture, and other models of that architecture (fine-tunes, other sizes) are expected
to run too. Measured on a MacBook Pro M4 Pro (24 GB), Chrome, 2026-10-13 and 14; every row rechecked in Chrome on the v0.5.1 build.

How each row is checked (references and replies in [`examples/chat/refs/`](../examples/chat/refs/)):
1. **Paging is exact**: llama.cpp natively, the same 9 conversations with paged experts against unpaged (or two slot
   counts when the whole model does not fit), greedy, 64 tokens: every reply must be identical.
2. **The browser path**: the same 9 conversations in Chrome (`scripts/chrome-gate.mjs`), compared with llama.cpp on
   Metal, or on the CPU where Metal cannot hold or load the model. Differences between GPUs come from rounding where two
   words score almost the same, so a match below 9/9 is expected; each miss was read.

| Architecture | Model checked (GGUF) | File | Paging exact | Chrome vs llama.cpp | Other models it covers |
|---|---|---|---|---|---|
| gemma4 | Gemma 4 26B-A4B, Q4_0 | 14.4 GB | 8 / 32 / 128 slots identical | 8/9 | Gemma 4 fine-tunes |
| qwen3moe | Qwen3 30B-A3B Instruct 2507, Q3_K_M | 14.7 GB | 9/9 (8 vs 32 slots) | 7/9 | Qwen3-Coder 30B, Qwen3 30B Thinking |
| qwen3next | Qwen3-Next 80B-A3B, Q4_K_M | 48.4 GB | 9/9 (16 vs 64 slots) | 5/9 vs CPU | Qwen3-Coder-Next (proof of concept: 2.9 tok/s) |
| qwen35moe | Qwen3.6 35B-A3B, UD-Q3_K_M | 16.6 GB | 9/9 (8 vs 32 slots) | 7/9 | Qwen3.5 35B-A3B and fine-tunes |
| qwen3vlmoe | Qwen3-VL 30B-A3B Instruct (text), Q3_K_M | 14.7 GB | 9/9 (8 vs 32 slots) | 7/9 | Qwen3-VL Thinking |
| deepseek2 | GLM-4.7-Flash, Q3_K_M | 14.6 GB | 9/9 (8 vs 32 slots) | 8/9 | DeepSeek-V2-Lite, Moonlight, Kimi-VL |
| nemotron_h_moe | Nemotron 3 Nano 30B-A3B, Q4_K_M | 22.4 GB | 9/9 (8 vs 32 slots) | 7/9 vs CPU | Nemotron 3.5 Lightning, Nemotron Cascade 2 |
| ernie4_5-moe | ERNIE 4.5 21B-A3B Thinking, Q4_K_M | 13.3 GB | 9/9 (8 vs 32 slots) | 7/9 | ERNIE 4.5 21B-A3B |
| gpt-oss | gpt-oss 20B, MXFP4 | 12.1 GB | 9/9 (unpaged vs 8) | 8/9 | gpt-oss-safeguard 20B |
| cohere2moe | North Mini Code 1.0, UD-Q3_K_M | 14.2 GB | 9/9 (8 vs 32 slots) | 5/9 | |
| lfm2moe | LFM2.5 24B-A2B, QAD-Q4_0 | 13.6 GB | 9/9 (8 vs 32 slots) | 7/9 | LFM2 8B-A1B (6/9), LFM2 24B-A2B |
| mellum | Mellum2 12B-A2.5B, Q4_K_M | 8.1 GB | 9/9 (unpaged vs 8) | 6/9 | Mellum2.1 Thinking |
| bailingmoe2 | Ling mini 2.0, Q4_K_M | 9.9 GB | 9/9 (unpaged vs 8) | 6/9 | Ring mini 2.0 |
| granitehybrid | Granite 4.0 H Tiny, Q4_K_M | 4.2 GB | 9/9 (unpaged vs 8) | 9/9 | |
| bailingmoe3 | Ling 3.0 tiny, Q4_K_M | 4.9 GB | 9/9 (unpaged vs 8) | 8/9 vs CPU | Edge0 8B-A1B |
| olmoe | OLMoE 1B-7B 0125 Instruct, Q4_K_M | 4.2 GB | 9/9 (unpaged vs 8) | 8/9 | |

**Runs, too slow to be practical**: Velvet-Eclipse 4x12B v0.2 (Mixtral-style `llama` MoE, Q3_K_M, 18.7 GB). With
experts paged at 2 slots, llama.cpp natively gives a correct reply at 0.74 tok/s, reading 1.4 GB from disk per token,
because it uses 2 of its 4 experts on every token. Not checked reply by reply: in Chrome no reply finished within 16
minutes. Paging helps when a token uses a small share of the experts (Qwen3.6: 8 of 256), not here.

**Runs, below the bar**: Hunyuan MT2 30B-A3B (hy_v3; paging exact 9/9, Chrome 3/9 against a CPU reference: coherent
replies that part at word choices, as GPUs and CPUs round differently) and Trinity Nano (afmoe; Chrome 3/9).

**Does not work yet**: Laguna XS 2.1 (laguna): llama.cpp's WebGPU backend returns unknown tokens for it, natively and in
Chrome, with or without paging; GigaChat 3.1 10B-A1.8B (deepseek2): empty replies in Chrome, though GLM-4.7-Flash on the
same architecture passes.

**Not covered**: dense models (no experts to page; a dense model runs only when it fits the GPU budget whole), maple
(ternary expert formats WebGPU lacks), diffusion models (paging is exact for diffusion MoE models such as LLaDA-MoE, but each denoising step reads nearly all
experts, about 3.9 GB per step, so they are not supported, for speed; llama.cpp's WebGPU output for LLaDA-MoE also degrades
where the CPU's does not, an open bug), and architectures llama.cpp does not support yet.
