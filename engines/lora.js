// Run-time LoRA for the native engines: a LoRA GGUF (llama.cpp's convert_lora_to_gguf.py format: per adapted matrix
// W, `W.lora_a` [N → R] and `W.lora_b` [R → M], F16) applied as y += scale · B(A·x) right after W's matmul, with
// scale = alpha / R (as llama.cpp's --lora) times the caller's scale. The model's weights stay as loaded, so an engine
// switches between adapters, or back to none, between turns with no reload (Qwen3MoeSsd.loadAdapter / setAdapter).
//
// An engine says which of its matmuls an adapter may follow by calling loraIo() as it binds them: the matrix's tensor
// name, its input and output buffers for one token, and (in initBatch) for a chunk. Adapters on the paged experts are
// refused: nothing reads them through a bound matmul.
import { GGML, parseGguf } from '../src/gguf.js';

const QB = 'struct Q { T: u32, pos0: u32, S: u32, pad: u32 }';
export const LORA_KERNELS = {
  // t = A·x, A [R × N] f16, one workgroup per rank row.
  loraA: `enable f16;
struct P { R: u32, N: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read_write> t: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let r = wg.x; let i = l.x; var acc = 0.0;
  for (var k = i; k < p.N; k += 256u) { acc += f32(a[r * p.N + k]) * x[k]; }
  red[i] = acc; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  if (i == 0u) { t[r] = red[0]; }
}`,
  // y += scale · B·t, B [M × R] f16, one thread per output row.
  loraB: `enable f16;
struct P { M: u32, R: u32, s: f32 }
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let m = g.x; if (m >= p.M) { return; }
  var acc = 0.0;
  for (var j = 0u; j < p.R; j++) { acc += f32(b[m * p.R + j]) * t[j]; }
  y[m] = y[m] + p.s * acc;
}`,
  // The chunk forms: t[tok] = A·x[tok], one workgroup per (rank row, token) ...
  loraAT: `enable f16;
struct P { R: u32, N: u32 }
${QB}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read_write> t: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> qd: Q;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let r = wg.x; let tok = wg.y; let i = l.x; if (tok >= qd.T) { return; }
  let xb = tok * p.N; var acc = 0.0;
  for (var k = i; k < p.N; k += 256u) { acc += f32(a[r * p.N + k]) * x[xb + k]; }
  red[i] = acc; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  if (i == 0u) { t[tok * p.R + r] = red[0]; }
}`,
  // ... and y[tok] += scale · B·t[tok], one thread per (output row, token).
  loraBT: `enable f16;
struct P { M: u32, R: u32, s: f32 }
${QB}
@group(0) @binding(0) var<storage, read> t: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> qd: Q;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let m = g.x; let tok = g.y; if (m >= p.M || tok >= qd.T) { return; }
  var acc = 0.0;
  for (var j = 0u; j < p.R; j++) { acc += f32(b[m * p.R + j]) * t[tok * p.R + j]; }
  y[tok * p.M + m] = y[tok * p.M + m] + p.s * acc;
}`,
};

const STORAGE = 0x80, COPY_SRC = 0x04, COPY_DST = 0x08;

// Reads a LoRA GGUF onto the engine's GPU and binds it to the matmuls the engine registered (engine.loraSites:
// tensor name → { layer, site, d: [x, y], bt: [x, y] | null }). Returns the adapter handle setAdapter() takes.
export async function loadLora(engine, source, { scale = 1 } = {}) {
  // a URL, a Blob (a File from OPFS or a picker), an ArrayBuffer or a Uint8Array
  const bytes = source instanceof Uint8Array ? source : new Uint8Array(typeof source === 'string'
    ? await (await fetch(source)).arrayBuffer() : source instanceof Blob ? await source.arrayBuffer() : source);
  const gguf = parseGguf(bytes), kv = gguf.kv;
  if (kv['adapter.type'] !== 'lora') throw new Error(`not a LoRA GGUF (adapter.type ${kv['adapter.type']})`);
  if (kv['general.architecture'] !== engine.arch) throw new Error(`adapter for ${kv['general.architecture']}, model is ${engine.arch}`);
  const byName = new Map(gguf.tensors.map((t) => [t.name, t]));
  const pairs = [];
  let R = 0, onGpu = 0;
  for (const ta of gguf.tensors) {
    if (!ta.name.endsWith('.lora_a')) continue;
    const base = ta.name.slice(0, -'.lora_a'.length), tb = byName.get(`${base}.lora_b`), io = engine.loraSites.get(base), w = engine.w[base];
    if (!tb) throw new Error(`${ta.name}: no matching lora_b`);
    if (!io || !w) throw new Error(`${base}: this engine has no run-time LoRA hook for it (adapters on the routed experts are not supported)`);
    if (engine.b && !io.bt) throw new Error(`${base}: no hook in this engine's chunked prefill`);
    const [N, r] = ta.dims, [r2, M] = tb.dims;
    if (r !== r2 || N !== w.cols || M !== w.rows) throw new Error(`${base}: adapter shape ${N}×${r} / ${r2}×${M} does not fit ${w.cols}×${w.rows}`);
    for (const t of [ta, tb]) if (t.type !== GGML.F16) throw new Error(`${t.name}: type ${t.type}, LoRA tensors must be F16`);
    R = Math.max(R, r);
    pairs.push({ io, ta, tb, r, N, M });
  }
  if (!pairs.length) throw new Error('the adapter has no lora_a / lora_b pairs');
  const upload = (t) => {
    const n = t.dims.reduce((x, y) => x * y, 1), buf = engine.buffer(n * 2, STORAGE | COPY_DST, 'adapter');
    engine.device.queue.writeBuffer(buf, 0, bytes, gguf.dataStart + t.offset, n * 2);
    onGpu += buf.size;
    return buf;
  };
  const s = (kv['adapter.lora.alpha'] ?? R) / R * scale;
  // R floats per token between the two halves, one buffer per adapter
  const lt = engine.buffer(R * 4, STORAGE | COPY_SRC | COPY_DST, 'adapter');
  const ltB = engine.b ? engine.buffer(engine.chunkTokens * R * 4, STORAGE | COPY_SRC | COPY_DST, 'adapter') : null;
  onGpu += lt.size + (ltB ? ltB.size : 0);
  const sites = new Map();
  for (const { io, ta, tb, r, N, M } of pairs) {
    const A = upload(ta), B = upload(tb), uA = engine.uniform([r, N]), uB = engine.uniform([M, r, { f: s }]);
    const e = { r, M, a: engine.bindR('loraA', [io.d[0], A, lt, uA]), b: engine.bindR('loraB', [lt, B, io.d[1], uB]) };
    if (io.bt && ltB) { e.aB = engine.bindR('loraAT', [io.bt[0], A, ltB, uA, engine.qb]); e.bB = engine.bindR('loraBT', [ltB, B, io.bt[1], uB, engine.qb]); }
    sites.set(`${io.layer}:${io.site}`, e);
  }
  return { name: kv['general.name'] || 'lora', rank: R, scale: s, matrices: pairs.length, bytes: onGpu, sites };
}
