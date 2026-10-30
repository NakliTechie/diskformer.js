/* clover_moe_ssd.js — Clover-35B-A3B-Preview (Diffuspace, Apache-2.0; GGUF arch `clover_moe`) in a browser tab,
 * larger than RAM. Clover is a block-diffusion model on Qwen3.6-35B-A3B: it fills a 32-token block of mask tokens
 * over several denoising passes, then commits the block into a causal prefix. The backbone, the expert paging and
 * the commit pass are Qwen35MoeSsd's (a commit is one prefillChunk); what is new is the preview pass:
 *   - DeltaNet layers: the 4-tap conv also looks right, inside the block (conv1d.weight_right, 3 taps), and the
 *     delta rule becomes a softmax over the block's 32 keys plus one "history" slot, whose value is the read-only
 *     clean recurrent state read with the query and decayed by the position's gate. Nothing is written back.
 *   - full-attention layers: every block token sees the whole prefix and the whole block (no causal mask). The
 *     block's K/V go into the cache rows after the prefix as scratch; a commit overwrites them.
 *   - the MoE: one shared expert subset per block (the coreset), the smallest set of experts, by summed routing mass
 *     of every token's top 8, that covers moe_block_coreset_top_p (0.9); each token then routes inside it.
 *   - logits come back for every block row, and row i scores slot i (not the next token).
 * The math follows the authors' reference (modeling_clover_moe.py: bd_gdn_conv1d, bd_gdn_local_memory, coreset_torch).
 * The denoising loop is not here: kohra's denoise module drives forward() below.
 *
 *   const m = await CloverMoeSsd.load(null, { url: '…/Clover-35B-A3B-Preview-Q4_0.gguf', poolBytes });
 *   await m.forward({ mode: 'commit', start: 0, ids: prompt, wantLogits: 'none' });
 *   const logits = await m.forward({ mode: 'preview', start: m.position, ids: block, wantLogits: 'all' });
 */

import { Qwen35MoeSsd } from './qwen35_moe_ssd.js';
import { STORAGE, COPY_SRC, COPY_DST, MAP_READ } from './qwen3_moe_ssd.js';
import { denoise, PRESETS } from './vendor/kohra-denoise.js';

// The authors' Q4_0 file: Q4_0 matrices, experts, embedding and output, F32 norms and scalars (the kernels
// Qwen35MoeSsd already has). Their Q8_0 file keeps the embedding, output and alpha/beta in BF16, which no kernel here reads.
export const CLOVER_35B_A3B = {
  repo: 'Diffuspace/Clover-35B-A3B-Preview-GGUF',
  file: 'Clover-35B-A3B-Preview-Q4_0.gguf',
  revision: '32e22a02b885e173177b7ce7661ad13448b4f89e',
  sha256: '3e2cf14ca47f6e41c251f08237ac79cd37ad81a2ea2a220fff731e34c66930d5',
  size: 19586909600,
};

const QB = 'struct Q { T: u32, pos0: u32, S: u32, pad: u32 }';
const NEG = '-3.4e38';

const PREVIEW_KERNELS = {
  // bd_gdn_conv1d: per channel, the causal 4-tap conv over [3 committed inputs | the block] plus the right taps
  // r[d-1]·x[t+d] (d = 1..3, inside the block), then SiLU. The committed inputs are only read.
  bdConvB: `
struct P { n: u32 }
${QB}
@group(0) @binding(0) var<storage, read> mixed: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> wr: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<uniform> qd: Q;
fn xin(c: u32, i: u32) -> f32 { if (i < 3u) { return st[c * 3u + i]; } return mixed[(i - 3u) * p.n + c]; }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let c = gi.x; if (c >= p.n) { return; }
  let w0 = w[c * 4u]; let w1 = w[c * 4u + 1u]; let w2 = w[c * 4u + 2u]; let w3 = w[c * 4u + 3u];
  for (var t = 0u; t < qd.T; t++) {
    var sum = 0.0;
    sum += xin(c, t) * w0; sum += xin(c, t + 1u) * w1; sum += xin(c, t + 2u) * w2; sum += xin(c, t + 3u) * w3;
    for (var d = 1u; d < 4u; d++) { if (t + d < qd.T) { sum += mixed[(t + d) * p.n + c] * wr[c * 3u + d - 1u]; } }
    y[t * p.n + c] = sum / (1.0 + exp(-sum));
  }
}`,
  // L2-normalised q and k (x / sqrt(Σx² + eps)) for (key head wg.x, token wg.y), into one buffer: q rows [T][kH·dS]
  // from 0, k rows from kOff (one buffer keeps bdGdnB within 10 storage bindings).
  bdQkNormB: `
struct P { kH: u32, dS: u32, keyDim: u32, convDim: u32, kOff: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> cv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qk: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> rq: array<f32, 128>;
var<workgroup> rk: array<f32, 128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let kh = wg.x; let t = wg.y; let j = l.x; let cb = t * p.convDim;
  let qv = cv[cb + kh * p.dS + j]; let kv = cv[cb + p.keyDim + kh * p.dS + j];
  rq[j] = qv * qv; rk[j] = kv * kv; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (j < s) { rq[j] += rq[j + s]; rk[j] += rk[j + s]; } workgroupBarrier(); }
  let o = t * p.keyDim + kh * p.dS + j;
  qk[o] = qv / sqrt(rq[0] + p.eps); qk[p.kOff + o] = kv / sqrt(rk[0] + p.eps);
}`,
  // bd_gdn_local_memory for (value head wg.x, query token wg.y); thread j = value row j. k-head = h % kH (the GGUF's
  // tiled head order, as gdnStep). Logits over the block's keys: (qn·kn)·logitScale[h]; the history slot's logit:
  // bias[h] + c[h]·log(T/32); its value: the clean state read with qn·scale, times exp(g), g = softplus(a + dt)·A.
  bdGdnB: `
struct P { kH: u32, dS: u32, keyDim: u32, convDim: u32, vH: u32, kOff: u32, scale: f32 }
${QB}
@group(0) @binding(0) var<storage, read> qk: array<f32>;
@group(0) @binding(1) var<storage, read> cv: array<f32>;
@group(0) @binding(2) var<storage, read> al: array<f32>;
@group(0) @binding(3) var<storage, read> dt: array<f32>;
@group(0) @binding(4) var<storage, read> aa: array<f32>;
@group(0) @binding(5) var<storage, read> ls: array<f32>;
@group(0) @binding(6) var<storage, read> hb: array<f32>;
@group(0) @binding(7) var<storage, read> hc: array<f32>;
@group(0) @binding(8) var<storage, read> M: array<f32>;
@group(0) @binding(9) var<storage, read_write> o: array<f32>;
@group(0) @binding(10) var<uniform> p: P;
@group(0) @binding(11) var<uniform> qd: Q;
var<workgroup> qs: array<f32, 128>;
var<workgroup> wt: array<f32, 33>;
fn log1p_(u: f32) -> f32 { let y = 1.0 + u; return log(y) - ((y - 1.0) - u) / y; }
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let h = wg.x; let t = wg.y; let j = l.x; let dS = p.dS; let kh = h % p.kH; let T = qd.T;
  let qb = t * p.keyDim + kh * dS;
  qs[j] = qk[qb + j];
  workgroupBarrier();
  if (j < T) {
    let kb = j * p.keyDim + kh * dS; var acc = 0.0;
    for (var i = 0u; i < dS; i++) { acc += qs[i] * qk[p.kOff + kb + i]; }
    wt[j] = acc * ls[h];
  }
  if (j == 32u) { wt[T] = hb[h] + hc[h] * log(f32(T) / 32.0); }
  workgroupBarrier();
  if (j == 0u) {
    var mx = wt[0];
    for (var s = 1u; s <= T; s++) { mx = max(mx, wt[s]); }
    var sum = 0.0;
    for (var s = 0u; s <= T; s++) { let e = exp(wt[s] - mx); wt[s] = e; sum += e; }
    for (var s = 0u; s <= T; s++) { wt[s] = wt[s] / sum; }
  }
  workgroupBarrier();
  let row = (h * dS + j) * dS; var rd = 0.0;
  for (var i = 0u; i < dS; i++) { rd += M[row + i] * (qs[i] * p.scale); }
  let x = al[t * p.vH + h] + dt[h];
  let sp = select(log1p_(exp(x)), x, x > 20.0);
  var acc = wt[T] * (rd * exp(sp * aa[h]));
  var loc = 0.0;
  for (var s = 0u; s < T; s++) { loc += wt[s] * cv[s * p.convDim + 2u * p.keyDim + h * dS + j]; }
  o[t * p.vH * dS + h * dS + j] = acc + loc;
}`,
  // Scores for T queries over all S = pos0 + T keys, unmasked: the block sees the prefix and itself both ways.
  attnScoreAllB: `enable f16;
struct P { heads: u32, kvHeads: u32, hd: u32, scale: f32 }
${QB}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f16>;
@group(0) @binding(2) var<storage, read_write> sc: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> qd: Q;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let idx = g.y * ng.x * 256u + g.x; let S = qd.S; let T = qd.T;
  if (idx >= p.heads * T * S) { return; }
  let h = idx / (T * S); let rem = idx % (T * S); let t = rem / S; let pos = rem % S;
  let kvh = h / (p.heads / p.kvHeads);
  let qo = t * p.heads * p.hd + h * p.hd; let ko = pos * p.kvHeads * p.hd + kvh * p.hd;
  var acc = 0.0;
  for (var d = 0u; d < p.hd; d++) { acc += q[qo + d] * f32(kc[ko + d]); }
  sc[idx] = acc * p.scale;
}`,
  // coreset_torch, one workgroup, thread e = expert (E ≤ 256). The block's mass on expert e: Σ over tokens and their
  // top-k picks of e of the pick's full-softmax probability; normalised, sorted (descending, ties by index), kept
  // while the mass before it is under topP (the first always). Router logits of every other expert become -3.4e38,
  // so topkB, run again, routes each token inside the subset (picks past the subset get weight 0).
  coresetB: `
struct P { E: u32, K: u32, topP: f32 }
${QB}
@group(0) @binding(0) var<storage, read_write> lg: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> qd: Q;
var<workgroup> red: array<f32, 256>;
var<workgroup> pr: array<f32, 256>;
var<workgroup> srt: array<f32, 256>;
var<workgroup> keep: array<u32, 256>;
var<workgroup> stat: array<f32, 2>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let e = l.x; let E = p.E; let on = e < E;
  var mine = 0.0;
  for (var t = 0u; t < qd.T; t++) {
    let v = select(${NEG}, lg[t * E + min(e, E - 1u)], on);
    red[e] = v; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (e < s) { red[e] = max(red[e], red[e + s]); } workgroupBarrier(); }
    if (e == 0u) { stat[0] = red[0]; } workgroupBarrier();
    let ex = select(0.0, exp(v - stat[0]), on);
    red[e] = ex; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (e < s) { red[e] += red[e + s]; } workgroupBarrier(); }
    if (e == 0u) { stat[1] = red[0]; } workgroupBarrier();
    for (var j = 0u; j < p.K; j++) { if (sel[t * 2u * p.K + j] == e) { mine += ex / stat[1]; } }
    workgroupBarrier();
  }
  red[e] = mine; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (e < s) { red[e] += red[e + s]; } workgroupBarrier(); }
  let tot = max(red[0], 1e-9);
  let pe = select(-1.0, mine / tot, on);
  pr[e] = pe; workgroupBarrier();
  var rank = 0u;
  for (var f = 0u; f < 256u; f++) { let pf = pr[f]; if (pf > pe || (pf == pe && f < e)) { rank++; } }
  srt[rank] = pe; workgroupBarrier();
  if (e == 0u) {
    var cum = 0.0;
    for (var r = 0u; r < E; r++) { keep[r] = select(0u, 1u, cum < p.topP || r == 0u); cum += srt[r]; }
  }
  workgroupBarrier();
  if (on && keep[rank] == 0u) { for (var t = 0u; t < qd.T; t++) { lg[t * E + e] = ${NEG}; } }
}`,
  // Per block row: the argmax with the mask id excluded (lowest id on a tie) and its softmax probability at T = 1
  // over the rest of the vocabulary: out[2t] = id, out[2t + 1] = probability bits. One workgroup per row.
  // A = -exp(A_log) in place: clover_moe GGUFs store ssm_a as A_log itself (checked against the BF16 checkpoint,
  // 2026-10-30), where the qwen35moe files store -exp(A_log), which gdnStep and bdGdnB read.
  negExp: `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gi: vec3<u32>) { if (gi.x < p.n) { a[gi.x] = -exp(a[gi.x]); } }`,
  top1B: `
struct P { V: u32, mask: u32 }
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let b = wg.x * p.V; let t = l.x;
  var bv = ${NEG}; var bi = 0xffffffffu;
  for (var i = t; i < p.V; i += 256u) { if (i != p.mask) { let v = lg[b + i]; if (v > bv) { bv = v; bi = i; } } }
  rv[t] = bv; ri[t] = bi; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { let o = rv[t + s]; let oi = ri[t + s]; if (o > rv[t] || (o == rv[t] && oi < ri[t])) { rv[t] = o; ri[t] = oi; } }
    workgroupBarrier();
  }
  let mx = rv[0]; let id = ri[0]; workgroupBarrier();
  var sum = 0.0;
  for (var i = t; i < p.V; i += 256u) { if (i != p.mask) { sum += exp(lg[b + i] - mx); } }
  rv[t] = sum; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { rv[t] += rv[t + s]; } workgroupBarrier(); }
  if (t == 0u) { out[2u * wg.x] = id; out[2u * wg.x + 1u] = bitcast<u32>(1.0 / rv[0]); }
}`,
};

export class CloverMoeSsd extends Qwen35MoeSsd {
  static archs = ['clover_moe'];

  static async load(modelId = null, opts = {}) {
    const source = opts.source || (opts.url ? { repo: null, file: opts.url.split('/').pop(), revision: null } : CLOVER_35B_A3B);
    return super.load(modelId, { ...opts, source });
  }

  constructor(device, manifest, gguf, opts) {
    super(device, manifest, gguf, opts);
    const kv = gguf.kv, g = (k) => kv[`clover_moe.${k}`];
    this.block = g('block_size') || 32;
    if (this.block > 32) throw new Error(`bdGdnB holds at most 32 block keys, got a block of ${this.block}`);
    this.coreset = g('moe_block_coreset_enabled') !== false;
    this.coresetTopP = g('moe_block_coreset_top_p') ?? 0.9;
    this.maskId = this.tokenizer.ids.get('<|mask|>');
    if (this.maskId == null) throw new Error('no <|mask|> token in the vocabulary');
    if (this.cfg.experts > 256) throw new Error('coresetB handles at most 256 experts');
    // capture: per-layer block rows of the last preview (test/browser/clover-states.html, the oracle gate)
    this.capture = !!opts.capture;
  }

  get kernels() {
    if (!this._ckernels) this._ckernels = { ...super.kernels, ...PREVIEW_KERNELS };
    return this._ckernels;
  }
  isGuarded(name) { return !(name in PREVIEW_KERNELS) && super.isGuarded(name); }

  async uploadDense(onProgress) {
    await super.uploadDense(onProgress);
    const s = this.cfg.ssm, enc = this.device.createCommandEncoder(), pass = enc.beginComputePass(), u = this.uniform([s.vHeads]);
    for (let l = 0; l < this.cfg.layers; l++) {
      if ((l + 1) % this.cfg.attnInterval === 0) continue;
      this.dispatch(pass, 'negExp', this.bind('negExp', [this.w[`blk.${l}.ssm_a`].raw, u]), Math.ceil(s.vHeads / 64));
    }
    pass.end(); this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  initBuffers() {
    super.initBuffers();
    if (!this.batchPrefill) throw new Error('CloverMoeSsd needs batchPrefill');
    if (this.chunkTokens < this.block) throw new Error(`a chunk of ${this.chunkTokens} tokens < block ${this.block}`);
    this.initPreview();
  }

  // Buffers and bind groups of the preview pass. It shares the chunk buffers (this.b) with prefillChunk: the two
  // never run at once (both hold the engine lock through forward()).
  initPreview() {
    const c = this.cfg, s = c.ssm, B = this.block, b = this.b, W = this.w, H = c.hidden;
    const keyDim = s.kHeads * s.dState, convDim = 2 * keyDim + s.dInner;
    const A = (n, cat = 'batch') => this.buffer(n * 4, STORAGE | COPY_SRC | COPY_DST, cat);
    this.p = { qk: A(2 * B * keyDim), logits: A(B * c.vocab), top1: A(2 * B) };
    this.rbTop1 = this.buffer(2 * B * 4, MAP_READ | COPY_DST, 'batch');
    this.rbBlockLogits = this.buffer(B * c.vocab * 4, MAP_READ | COPY_DST, 'batch');
    if (this.capture) { this.p.cap = A(c.layers * B * H); this.rbCap = this.buffer(c.layers * B * H * 4, MAP_READ | COPY_DST, 'batch'); }
    const U = (w) => this.uniform(w), qb = this.qb, raw = (name) => {
      const t = W[name]; if (!t || !t.raw) throw new Error(`missing F32 tensor ${name}`); return t.raw;
    };
    const u = {
      conv: U([convDim]), qkn: U([s.kHeads, s.dState, keyDim, convDim, B * keyDim, { f: c.eps }]),
      gdn: U([s.kHeads, s.dState, keyDim, convDim, s.vHeads, B * keyDim, { f: 1 / Math.sqrt(s.dState) }]),
      att: U([c.heads, c.kvHeads, c.headDim, { f: 1 / Math.sqrt(c.headDim) }]),
      core: U([c.experts, c.topK, { f: this.coresetTopP }]), lm: U([c.vocab, H]), top1: U([c.vocab, this.maskId]),
    };
    for (let l = 0; l < c.layers; l++) {
      const n = (t) => `blk.${l}.${t}`, g = {};
      if (this.recurrent[l]) {
        g.conv = this.bind('bdConvB', [b.mixed, raw(n('ssm_conv1d.weight')), raw(n('ssm_conv1d.weight_right')), this.conv[l], b.cv, u.conv, qb]);
        g.qkn = this.bind('bdQkNormB', [b.cv, this.p.qk, u.qkn]);
        g.gdn = this.bind('bdGdnB', [this.p.qk, b.cv, b.al, raw(n('ssm_dt.bias')), raw(n('ssm_a')),
          raw(n('ssm_bd_logit_scale.weight')), raw(n('ssm_bd_history.bias')), raw(n('ssm_bd_history.weight')), this.ssm[l], b.go, u.gdn, qb]);
      } else {
        g.score = this.bind('attnScoreAllB', [b.q, this.kc[l], b.sc, u.att, qb]);
      }
      g.core = this.bind('coresetB', [b.rl, b.sel, u.core, qb]);
      this.layers[l].P = g;
    }
    this.gP = {
      rmsOut: this.bind('rmsnormB', [b.x, raw('output_norm.weight'), b.xn, U([H, { f: c.eps }])]),
      lm: this.bind(this.qk.mmT, [b.xn, W['output.weight'].q, W['output.weight'].s, this.p.logits, u.lm, qb]),
      top1: this.bind('top1B', [this.p.logits, this.p.top1, u.top1]),
    };
    if (this.qk.mmT !== 'matmulQ4T' && this.qk.mmT !== 'matmulQ8T') throw new Error('the batched LM head needs a Q4_0 or Q8_0 output matrix');
    if (W['output.weight'].q == null) throw new Error('the batched LM head needs a Q4_0 or Q8_0 output matrix');
  }

  // Layer l of the preview over the block's T rows, up to its routing (encodeTrunkB with the preview kernels).
  encodePreviewTrunk(pass, l, T, S) {
    const c = this.cfg, s = c.ssm, g = this.layers[l].B, gp = this.layers[l].P, d = (nm, gr, x, y, z) => this.dispatch(pass, nm, gr, x, y, z);
    const H = c.hidden, QN = c.heads * c.headDim, kvn = c.kvHeads * c.headDim, convDim = 2 * s.kHeads * s.dState + s.dInner;
    const tt = Math.ceil(T / 16), mm = (n) => Math.ceil(n / 16), QK = this.qk;
    d('rmsnormB', g.rmsA, T);
    if (this.recurrent[l]) {
      d(QK.mmT, g.qkv, mm(convDim), tt); d(QK.mmT, g.z, mm(s.dInner), tt); d(QK.mmT, g.alpha, mm(s.vHeads), tt);
      d('bdConvB', gp.conv, Math.ceil(convDim / 256));
      d('bdQkNormB', gp.qkn, s.kHeads, T);
      d('bdGdnB', gp.gdn, s.vHeads, T);
      d('gdnNormGateB', g.norm, s.vHeads, T);
      d(QK.mmT, g.out, mm(H), tt);
    } else {
      d(QK.mmT, g.q, mm(2 * QN), tt); d(QK.mmT, g.k, mm(kvn), tt); d(QK.mmT, g.v, mm(kvn), tt);
      d('qkNormRopePB', g.ropeQ, c.heads, T); d('qkNormRopePB', g.ropeK, c.kvHeads, T);
      d('kvStoreB', g.kv, Math.ceil(kvn / 256), T);   // scratch rows pos0.. after the prefix
      d('attnScoreAllB', gp.score, Math.ceil(c.heads * T * S / 256));
      d('softmaxB', g.soft, c.heads * T);
      d('attnOutPosB', g.attOut, Math.ceil(QN / 256), T);
      d('sigmoidMulB', g.gate, Math.ceil(QN / 256), T);
      d(QK.mmT, g.o, mm(H), tt);
    }
    d('addInPlaceB', g.addO, Math.ceil(H / 256), T);
    d('rmsnormB', g.rmsF, T);
    d('matmulF32T', g.router, c.experts, T);
    d('topkB', g.topk, T);
    if (this.coreset) { d('coresetB', gp.core, 1); d('topkB', g.topk, T); }
    d(QK.mmT, g.shg, mm(c.shexpFf), tt); d(QK.mmT, g.shu, mm(c.shexpFf), tt);
    d('siluMul2B', g.shact, Math.ceil(c.shexpFf / 256), T);
    d(QK.mmT, g.shd, mm(H), tt);
    d('matmulF32T', g.shgate, 1, T);
  }

  // One denoising pass over the block `ids` (mask slots hold this.maskId) at positions position..position+T-1.
  // Reads the committed cache and writes nothing to it (KV rows past the prefix are scratch). want: 'all' → T × vocab
  // logits (Float32Array, row t scores slot t); 'top1' → { token: Int32Array(T), conf: Float32Array(T) }, the argmax
  // over the vocabulary without the mask id and its probability at T = 1.
  async preview(ids, want = 'all') {
    const c = this.cfg, dev = this.device, K = c.topK, H = c.hidden, T = ids.length, pos0 = this.position, S = pos0 + T;
    if (T < 1 || T > this.block || this.block % T) throw new Error(`a preview block of ${T} must divide ${this.block}`);
    if (S > this.maxCtx) throw this.contextFull();
    const tStart = performance.now();
    dev.queue.writeBuffer(this.a.stop, 0, new Uint32Array(4));
    dev.queue.writeBuffer(this.b.ids, 0, Uint32Array.from(ids));
    dev.queue.writeBuffer(this.qb, 0, new Uint32Array([T, pos0, S, 0]));
    dev.queue.writeBuffer(this.b.rope, 0, this.ropeRows(this.textPositions(T), T));
    const selBytes = T * 2 * K * 4, rows = T * H * 4;
    let enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    this.dispatch(pass, this.qk.embB, this.gB.embed, Math.ceil(H / 256), T);
    this.encodePreviewTrunk(pass, 0, T, S);
    pass.end();
    enc.copyBufferToBuffer(this.b.sel, 0, this.rbSelB, 0, selBytes);
    dev.queue.submit([enc.finish()]);
    const units = new Set();
    for (let l = 0; l < c.layers; l++) {
      const t0 = performance.now();
      await this.rbSelB.mapAsync(1, 0, selBytes);
      this.counters.gpuWaitMs += performance.now() - t0;
      const sel = new Uint32Array(this.rbSelB.getMappedRange(0, selBytes).slice(0)), wts = new Float32Array(sel.buffer);
      this.rbSelB.unmap();
      // Picks with weight 0 (a token routed past the coreset) need no load: they take any loaded slot.
      const pairs = new Uint32Array(T * K), live = new Set();
      for (let t = 0; t < T; t++) for (let j = 0; j < K; j++) {
        const e = sel[t * 2 * K + j]; pairs[t * K + j] = e;
        if (wts[t * 2 * K + K + j] > 0) live.add(e);
      }
      const unique = [...live];
      for (const e of unique) units.add(l * c.experts + e);
      const tq = performance.now();
      const slotsU = await this.xs.ensure(l, unique);
      this.counters.ensureMs += performance.now() - tq;
      const slotOf = new Map(unique.map((e, i) => [e, slotsU[i]]));
      dev.queue.writeBuffer(this.b.slots, 0, pairs.map((e) => slotOf.get(e) ?? slotsU[0]));
      enc = dev.createCommandEncoder(); pass = enc.beginComputePass();
      this.encodeExpertsB(pass, l, T);
      pass.end();
      if (this.capture) enc.copyBufferToBuffer(this.b.x, 0, this.p.cap, l * this.block * H * 4, rows);
      if (l + 1 < c.layers) {
        pass = enc.beginComputePass();
        this.encodePreviewTrunk(pass, l + 1, T, S);
        pass.end();
        enc.copyBufferToBuffer(this.b.sel, 0, this.rbSelB, 0, selBytes);
      } else {
        pass = enc.beginComputePass();
        this.dispatch(pass, 'rmsnormB', this.gP.rmsOut, T);
        this.dispatch(pass, this.qk.mmT, this.gP.lm, Math.ceil(c.vocab / 16), Math.ceil(T / 16));
        if (want === 'top1') this.dispatch(pass, 'top1B', this.gP.top1, T);
        pass.end();
        if (want === 'top1') enc.copyBufferToBuffer(this.p.top1, 0, this.rbTop1, 0, 2 * T * 4);
        else enc.copyBufferToBuffer(this.p.logits, 0, this.rbBlockLogits, 0, T * c.vocab * 4);
      }
      dev.queue.submit([enc.finish()]);
      this.xs.release(slotsU);
    }
    let result;
    if (want === 'top1') {
      await this.rbTop1.mapAsync(1, 0, 2 * T * 4);
      const u = new Uint32Array(this.rbTop1.getMappedRange(0, 2 * T * 4).slice(0)), f = new Float32Array(u.buffer);
      this.rbTop1.unmap();
      result = { token: new Int32Array(T), conf: new Float32Array(T) };
      for (let t = 0; t < T; t++) { result.token[t] = u[2 * t]; result.conf[t] = f[2 * t + 1]; }
    } else {
      await this.rbBlockLogits.mapAsync(1, 0, T * c.vocab * 4);
      result = new Float32Array(this.rbBlockLogits.getMappedRange(0, T * c.vocab * 4).slice(0));
      this.rbBlockLogits.unmap();
    }
    this.counters.previews = (this.counters.previews || 0) + 1;
    this.counters.previewExperts = (this.counters.previewExperts || 0) + units.size;
    this.counters.previewMs = (this.counters.previewMs || 0) + performance.now() - tStart;
    return result;
  }

  // The last preview's per-layer residual rows: layers × T × hidden (capture: true only).
  async readCapture(T = this.block) {
    const c = this.cfg, n = c.layers * this.block * c.hidden * 4, enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.p.cap, 0, this.rbCap, 0, n);
    this.device.queue.submit([enc.finish()]);
    await this.rbCap.mapAsync(1);
    const all = new Float32Array(this.rbCap.getMappedRange().slice(0));
    this.rbCap.unmap();
    return Array.from({ length: c.layers }, (_, l) => all.subarray(l * this.block * c.hidden, (l * this.block + T) * c.hidden));
  }

  // The callback kohra's denoise module drives (one call at a time, under the engine lock):
  //   commit:  appends ids at positions [start, start + ids.length); start is the cache length, or 0 for a new prompt
  //            (which reuses the cached prefix it shares). Returns null (Clover commits never need logits).
  //   preview: a denoising pass over the block at the cache end; see preview().
  async forward(args) {
    const release = await this.acquire();
    try { return await this.forwardLocked(args); } catch (e) { throw this.stepError(e); } finally { release(); }
  }
  async forwardLocked({ mode, start, ids, wantLogits = mode === 'preview' ? 'all' : 'none' }) {
    ids = Array.from(ids);
    if (mode === 'preview') {
      if (start !== this.position) throw new Error(`preview at ${start}, but the cache holds ${this.position} tokens`);
      if (wantLogits !== 'all' && wantLogits !== 'top1') throw new Error(`preview returns 'all' or 'top1', not ${wantLogits}`);
      return await this.preview(ids, wantLogits);
    }
    if (mode !== 'commit') throw new Error(`forward mode ${mode}`);
    if (wantLogits !== 'none') throw new Error('Clover commits return no logits');
    if (start === 0) { await this.prefillSegments([{ ids }], 'none', { sysEnd: this.checkpointEnd(ids) }); return null; }
    if (start !== this.position) throw new Error(`commit at ${start}, but the cache holds ${this.position} tokens`);
    for (let i = 0; i < ids.length; i += this.chunkTokens) {
      const chunk = ids.slice(i, i + this.chunkTokens);
      if (chunk.length < 2) await this.step(chunk[0], 'none'); else await this.prefillChunk(chunk, 'none');
    }
    return null;
  }

  // A chat turn (generate() holds the engine lock around it): kohra's denoise loop with Clover's decoding rules
  // (PRESETS.clover: threshold 0.95, the reference generation config's penalties), yielding text block by block.
  // opts beyond maxNewTokens / signal / enableThinking go to denoise() (e.g. penalties: null, top1: true).
  async *generateTurn(messages, { maxNewTokens = 512, signal, enableThinking = true, ...loop } = {}) {
    const promptIds = this.tokenizer.encode(this.chatPrompt(messages, { enableThinking }));
    const blocks = [], out = [];
    let wake = null, done = false, failed = null, stop = false;
    const STOP = new Error('stopped');
    const forward = (a) => { if (stop || (signal && signal.aborted)) throw STOP; return this.forwardLocked(a); };
    const run = denoise({ ...PRESETS.clover, top1: true, ...loop, forward, promptIds, maxNewTokens, maskId: this.maskId, blockSize: this.block,
      onBlock: ({ tokens }) => { blocks.push(tokens); if (wake) wake(); } })
      .then((r) => { this.lastDenoise = { previews: r.previews, commits: r.commits, seconds: r.seconds, tokens: r.tokenIds.length }; },
        (e) => { if (e !== STOP) failed = e; })
      .finally(() => { done = true; if (wake) wake(); });
    try {
      for (;;) {
        while (blocks.length) {
          for (const token of blocks.shift()) { out.push(token); yield { text: this.tokenizer.decode(out), token, tokens: out.length }; }
        }
        if (done) break;
        await new Promise((r) => { wake = r; });
        wake = null;
      }
      if (failed) throw failed;
    } finally { stop = true; await run; }
  }

  stats() {
    const n = Math.max(1, this.counters.previews || 0);
    return { ...super.stats(), previews: this.counters.previews || 0, previewMsEach: (this.counters.previewMs || 0) / n, expertsPerPreview: (this.counters.previewExperts || 0) / n };
  }
}

export { PREVIEW_KERNELS as CLOVER_PREVIEW_KERNELS };
