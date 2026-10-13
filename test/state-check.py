#!/usr/bin/env python3
"""state-check.py: compare the engine's per-layer states (test/browser/states.html) with a numpy forward pass of
llama.cpp's graph (LocalMind's scripts/gemma4-ref.py), layer by layer, at the prompt's last position.

  python3 test/state-check.py ENGINE.json REF.npz [--tol 1e-4]

Relative error per layer = max |engine - numpy| / max |numpy|. Fails above --tol. The 2026-10-05 gate measured
<= 2e-5 on the 6-layer cut.
"""
import json, sys
import numpy as np

eng = json.load(open(sys.argv[1]))
ref = np.load(sys.argv[2])
tol = float(sys.argv[sys.argv.index('--tol') + 1]) if '--tol' in sys.argv else 1e-4
assert eng.get('ok'), eng.get('error')
assert list(ref['ids']) == eng['ids'], 'the engine and numpy ran different prompts'
worst = 0.0
for name, key in (('attn_out', 'attnOut'), ('layer', 'layers')):
    for l, ours in enumerate(eng[key]):
        want = ref['attn_out' if name == 'attn_out' else 'layers'][l][-1]
        err = float(np.abs(np.asarray(ours, np.float32) - want).max() / np.abs(want).max())
        worst = max(worst, err)
        print(f'{name} {l}: relative error {err:.2e}')
print(f'worst {worst:.2e}, tolerance {tol:.0e}: {"ok" if worst <= tol else "FAIL"}')
sys.exit(0 if worst <= tol else 1)
