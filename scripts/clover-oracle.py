#!/usr/bin/env python3
"""clover-oracle.py: the reference for the clover_moe engine's preview and commit passes, on CPU, layer by layer.

Runs the authors' own layer classes (modeling_clover_moe.py from Diffuspace/Clover-35B-A3B-Preview, unmodified) in
float32 with weights dequantized from a layer cut of the same GGUF the engine reads (scripts/gguf-cut.py), so the
engine and the oracle differ only in arithmetic order. The GGUF stores value heads in llama.cpp's tiled order and
zero-centred RMSNorm weights as 1 + w; both are undone here. Unlike the
qwen35moe files, ssm_a holds A_log itself (checked against the BF16 checkpoint) to feed the reference its own layout.

The sequence (test/refs/clover-oracle-ids.json, which test/browser/clover-states.html replays):
  commit the prompt → preview A (6 real tokens, 26 masks) → commit block C (32 tokens) → preview B (32 masks)

  python3 scripts/clover-oracle.py CUT.gguf --ref-dir <folder with modeling_clover_moe.py, tokenizer.json> --out REF.npz
"""
import importlib, json, math, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
cut, ref_dir, out = sys.argv[1], Path(arg('--ref-dir')), arg('--out')
ids_path = Path(__file__).resolve().parent.parent / 'test/refs/clover-oracle-ids.json'
torch.set_grad_enabled(False)

# The reference modules as a package (they import each other relatively).
pkg = Path(tempfile.mkdtemp()) / 'clover_ref'
pkg.mkdir()
for f in ('modeling_clover_moe.py', 'sampling_clover_moe.py', 'configuration_clover_moe.py'):
    shutil.copy(ref_dir / f, pkg / f)
(pkg / '__init__.py').write_text('')
sys.path.insert(0, str(pkg.parent))
M = importlib.import_module('clover_ref.modeling_clover_moe')

r = GGUFReader(cut)
T = {t.name: t for t in r.tensors}
L = int(r.fields['clover_moe.block_count'].parts[-1][0])
hf = json.load(open(ref_dir / 'config.json'))
cfg = types.SimpleNamespace(**{**hf, 'num_hidden_layers': L, 'layer_types': hf['layer_types'][:L], 'attention_backend': 'eager'})

def deq(name):
    t = T[name]
    return torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type), dtype=np.float32))

# GGUF value head g = r·kH + kh holds the reference's head kh·(vH/kH) + r.
vH, kH, dS = cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_value_head_dim
rep = vH // kH
g_of = torch.tensor([(i % rep) * kH + i // rep for i in range(vH)])
heads = lambda x: x.reshape(vH, -1, *x.shape[1:])[g_of].reshape(x.shape)
def qkv_rows(x):
    kd = 2 * kH * cfg.linear_key_head_dim
    return torch.cat((x[:kd], heads(x[kd:])))

def load_layer(i):
    layer = M.CloverDecoderLayer(cfg, i).float()
    n = lambda s: f'blk.{i}.{s}'
    p = {'input_layernorm.weight': deq(n('attn_norm.weight')) - 1, 'post_attention_layernorm.weight': deq(n('post_attention_norm.weight')) - 1,
         'mlp.gate.weight': deq(n('ffn_gate_inp.weight')), 'mlp.shared_expert.gate_proj.weight': deq(n('ffn_gate_shexp.weight')),
         'mlp.shared_expert.up_proj.weight': deq(n('ffn_up_shexp.weight')), 'mlp.shared_expert.down_proj.weight': deq(n('ffn_down_shexp.weight')),
         'mlp.shared_expert_gate.weight': deq(n('ffn_gate_inp_shexp.weight')).reshape(1, -1)}
    if layer.preview_type == 'linear_attention':
        a = 'linear_attn.'
        p.update({a + 'in_proj_qkv.weight': qkv_rows(deq(n('attn_qkv.weight'))), a + 'in_proj_z.weight': heads(deq(n('attn_gate.weight'))),
                  a + 'in_proj_a.weight': heads(deq(n('ssm_alpha.weight'))), a + 'in_proj_b.weight': heads(deq(n('ssm_beta.weight'))),
                  a + 'A_log': heads(deq(n('ssm_a'))), a + 'dt_bias': heads(deq(n('ssm_dt.bias'))),
                  a + 'conv1d.weight': qkv_rows(deq(n('ssm_conv1d.weight'))).reshape(-1, 1, 4),
                  a + 'conv_right_weight': qkv_rows(deq(n('ssm_conv1d.weight_right'))).reshape(-1, 1, 3),
                  a + 'bd_softmax_logit_scale': heads(deq(n('ssm_bd_logit_scale.weight'))),
                  a + 'bd_softmax_history_bias': heads(deq(n('ssm_bd_history.bias'))),
                  a + 'bd_softmax_history_c': heads(deq(n('ssm_bd_history.weight'))),
                  a + 'norm.weight': deq(n('ssm_norm.weight')),
                  a + 'out_proj.weight': deq(n('ssm_out.weight')).reshape(-1, vH, dS)[:, g_of].reshape(cfg.hidden_size, -1)})
    else:
        a = 'self_attn.'
        p.update({a + 'q_proj.weight': deq(n('attn_q.weight')), a + 'k_proj.weight': deq(n('attn_k.weight')), a + 'v_proj.weight': deq(n('attn_v.weight')),
                  a + 'o_proj.weight': deq(n('attn_output.weight')), a + 'q_norm.weight': deq(n('attn_q_norm.weight')) - 1,
                  a + 'k_norm.weight': deq(n('attn_k_norm.weight')) - 1})
    missing = set(k for k, _ in layer.named_parameters() if not k.startswith('mlp.experts.')) - set(p)
    assert not missing, missing
    for k, v in p.items():
        dst = layer.get_parameter(k)
        assert dst.shape == v.shape, (k, dst.shape, v.shape)
        dst.copy_(v)
    layer.mlp.experts = LazyExperts(i)
    return layer

class LazyExperts(torch.nn.Module):
    """CloverExperts.forward, with each expert's weights dequantized from the GGUF on first use."""
    def __init__(self, i):
        super().__init__()
        self.t = [T[f'blk.{i}.ffn_{w}_exps.weight'] for w in ('gate', 'up', 'down')]
        self.cache = {}
    def w(self, e):
        if e not in self.cache:
            g, u, d = (torch.from_numpy(np.ascontiguousarray(dequantize(t.data[e], t.tensor_type), dtype=np.float32)) for t in self.t)
            self.cache[e] = (torch.cat((g, u)), d)
        return self.cache[e]
    def forward(self, x, ids, weights):
        out = torch.zeros_like(x, dtype=torch.float32)
        for expert in ids.unique().tolist():
            token, slot = torch.where(ids == expert)
            gu, dn = self.w(expert)
            gate, up = torch.nn.functional.linear(x[token], gu).chunk(2, -1)
            hidden = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
            y = M.linear_fp32_output(hidden, dn)
            routed = (y * weights[token, slot, None].float()).to(x.dtype)
            out.index_add_(0, token, routed.float())
        return out.to(x.dtype)

layers = [load_layer(i) for i in range(L)]
norm = M.CloverRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).float()
norm.weight.copy_(deq('output_norm.weight') - 1)
emb_t, out_t = T['token_embd.weight'], T['output.weight']
embed = lambda ids: torch.from_numpy(np.ascontiguousarray(dequantize(emb_t.data[ids], emb_t.tensor_type), dtype=np.float32))
def lm_head(x):
    rows = out_t.data.shape[0]
    return torch.cat([x @ torch.from_numpy(np.ascontiguousarray(dequantize(out_t.data[o:o + 16384], out_t.tensor_type), dtype=np.float32)).T
                      for o in range(0, rows, 16384)], -1)

def run(ids, cache, preview):
    """CloverMoeModel.forward + lm_head (batch 1, every token valid), with each layer's output kept."""
    x = embed(np.asarray(ids))[None]
    t = x.shape[1]
    prefix_valid = cache.attention_mask if cache is not None else torch.zeros((1, 0), dtype=torch.bool)
    valid = torch.ones((1, t), dtype=torch.bool)
    positions = prefix_valid.sum(-1, keepdim=True) + valid.long().cumsum(-1) - 1
    states, caps = [], []
    for i, layer in enumerate(layers):
        x, st = layer(x, cache.layers[i] if cache is not None else None, valid, prefix_valid, positions, preview)
        states.append(st)
        caps.append(x[0].clone())
    nxt = cache if preview else M.CloverCache(states, torch.cat((prefix_valid, valid), 1))
    logits = lm_head(norm(x))[0] if preview else None
    return nxt, torch.stack(caps), logits

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(ref_dir / 'tokenizer.json'))
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
MASK = hf['mask_token_id']
prompt = enc('<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n')
a_real = enc('The user asks about the capital')[:6]
block_a = a_real + [MASK] * (32 - len(a_real))
block_c = enc('The user asks about the capital of France. The answer is Paris, which has been the capital for centuries and is its largest city by far. It sits on the Seine in the north of the country.')[:32]
assert len(block_c) == 32, len(block_c)
block_b = [MASK] * 32
json.dump({'prompt': prompt, 'previewA': block_a, 'commitC': block_c, 'previewB': block_b, 'mask': MASK}, open(ids_path, 'w'))

cache, _, _ = run(prompt, None, False)
_, capA, logA = run(block_a, cache, True)
cache, _, _ = run(block_c, cache, False)
_, capB, logB = run(block_b, cache, True)
np.savez(out, layersA=capA.numpy(), logitsA=logA.numpy(), layersB=capB.numpy(), logitsB=logB.numpy())
for name, lg in (('A', logA), ('B', logB)):
    lg = lg.clone(); lg[:, MASK] = -math.inf
    p = lg.softmax(-1).max(-1)
    print(f'preview {name}: argmax {p.indices[:8].tolist()} … conf {[round(v, 3) for v in p.values[:8].tolist()]} …')
print(f'wrote {out} and {ids_path}')
