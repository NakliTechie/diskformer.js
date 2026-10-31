# A random LoRA GGUF for a model, in llama.cpp's adapter format (convert_lora_to_gguf.py's: `W.lora_a` [N -> R] and
# `W.lora_b` [R -> M], F16, adapter.type = lora, adapter.lora.alpha). For gating run-time LoRA in an engine against
# `llama-server --lora` with the same file, on models that have no published adapter: the comparison checks the
# arithmetic and which matrices it follows, not what an adapter learned. Seeded, so the same arguments give the same
# file.
#   PYTHONPATH=<llama.cpp>/gguf-py python3 scripts/make-test-lora.py <base.gguf> <out.gguf> --targets attn_q,attn_k,attn_v,attn_output [--rank 32]
#       [--alpha 64] [--sigma 0.004] [--seed 0]
# --targets: tensor-name parts after `blk.N.` and before `.weight`; every layer that has one is adapted.
# --sigma: the std of B's entries (A's is 1/sqrt(N)); the change to a matmul's output is about
#   alpha/R * sigma * sqrt(R) of its input's RMS.
import argparse
import re

import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('out')
ap.add_argument('--targets', required=True)
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--alpha', type=float, default=64)
ap.add_argument('--sigma', type=float, default=0.004)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

r = gguf.GGUFReader(a.base)
arch = r.fields['general.architecture'].contents()
targets = set(a.targets.split(','))
rng = np.random.default_rng(a.seed)
w = gguf.GGUFWriter(a.out, arch)
w.add_string('general.type', 'adapter')
w.add_string('general.name', f'random rank-{a.rank} test LoRA (seed {a.seed})')
w.add_string('adapter.type', 'lora')
w.add_float32('adapter.lora.alpha', a.alpha)
n = 0
for t in r.tensors:
    m = re.fullmatch(r'blk\.(\d+)\.(.+)\.weight', t.name)
    if not m or m.group(2) not in targets or len(t.shape) != 2:
        continue
    N, M = int(t.shape[0]), int(t.shape[1])   # ne0 = input width, ne1 = output rows
    A = (rng.standard_normal((a.rank, N)) / np.sqrt(N)).astype(np.float16)
    B = (rng.standard_normal((M, a.rank)) * a.sigma).astype(np.float16)
    w.add_tensor(f'{t.name}.lora_a', A)       # numpy (R, N) -> ne [N, R]
    w.add_tensor(f'{t.name}.lora_b', B)       # numpy (M, R) -> ne [R, M]
    n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f'{a.out}: {n} adapted matrices ({arch}), rank {a.rank}, alpha {a.alpha}, sigma {a.sigma}, seed {a.seed}')
