#!/usr/bin/env python3
"""clover-check.py: compare the clover_moe engine (test/browser/clover-states.html) with the torch oracle
(scripts/clover-oracle.py) on a layer cut: per-layer residual rows of both previews, the logits at every 97th id,
and each row's top-1 (the engine's 'all' logits and its GPU 'top1' path).

  python3 scripts/clover-check.py ENGINE.json REF.npz [--tol 1e-3]

Relative error = max |engine - torch| / max |torch|, per layer over all 32 rows. Fails above --tol, or when a row's
top-1 differs where the oracle's top two logits are more than 1e-3 apart (a near tie may flip in f32).
"""
import json, sys
import numpy as np
eng = json.load(open(sys.argv[1])); ref = np.load(sys.argv[2])
tol = float(sys.argv[sys.argv.index('--tol') + 1]) if '--tol' in sys.argv else 1e-3
assert eng.get('ok'), eng.get('error')
worst, bad = 0.0, 0
for b in ('A', 'B'):
    want = ref[f'layers{b}']
    for l, ours in enumerate(eng[b]['layers']):
        ours = np.asarray(ours, np.float32).reshape(want[l].shape)
        err = float(np.abs(ours - want[l]).max() / np.abs(want[l]).max()); err = err if np.isfinite(err) else float('inf'); worst = max(worst, err)
        print(f'preview {b} layer {l}: relative error {err:.2e}')
    lg = ref[f'logits{b}'].copy()
    sub = lg[:, ::97]
    err = float(np.abs(np.asarray(eng[b]['logitsSub'], np.float32).reshape(sub.shape) - sub).max() / np.abs(sub).max()); err = err if np.isfinite(err) else float('inf'); worst = max(worst, err)
    print(f'preview {b} logits: relative error {err:.2e}')
    lg[:, eng['mask']] = -np.inf
    order = np.argsort(-lg, -1)[:, :2]
    for t in range(lg.shape[0]):
        a, s = order[t]; gap = lg[t, a] - lg[t, s]
        p = 1 / np.exp(lg[t] - lg[t, a]).sum()
        for name, (tk, cf) in (('all', eng[b]['top'][t]), ('top1', (eng[b]['top1']['token'][t], eng[b]['top1']['conf'][t]))):
            if tk != a and gap > 1e-3:
                bad += 1; print(f'  preview {b} row {t} {name}: token {tk}, oracle {a} (gap {gap:.2e})')
            if abs(cf - p) > max(1e-3, 1e-2 * p):
                bad += 1; print(f'  preview {b} row {t} {name}: conf {cf:.5f}, oracle {p:.5f}')
worst = worst if np.isfinite(worst) else float('inf')
print(f'worst {worst:.2e}, tolerance {tol:.0e}, top-1 mismatches {bad}: {"ok" if worst <= tol and not bad else "FAIL"}')
sys.exit(0 if worst <= tol and not bad else 1)
