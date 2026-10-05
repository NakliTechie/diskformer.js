// The whole loop an engine runs with diskformer, end to end:
//   1. once: write the table to OPFS (RowFile.write) — here a synthetic float table, in an engine the model's bytes
//   2. once: create the GPU buffer that holds `SLOTS` rows and hand it to a RowCache
//   3. every step: cache.lookup(ids) → slots; write the slots; dispatch a kernel that reads row slot × D
// RowFile needs OPFS sync access handles, which exist only in dedicated workers, so this runs in one.
import { RowFile, RowCache, fingerprint } from '../../index.js';

const V = 131072, D = 256, ROW_BYTES = D * 4;          // 131,072 tokens × 256 f32 = 128 MB on disk
const SLOTS = 1024, STEPS = 200, BATCH = 32;           // 1 MB of rows on the GPU; 32 ids per step
const value = (t, d) => Math.fround(Math.sin(t * 0.37 + d * 0.11));   // the "model": any deterministic table
const log = (s) => self.postMessage({ log: s });

const GATHER = `
struct P { D: u32, n: u32 }
@group(0) @binding(0) var<storage, read> rows: array<f32>;       // the cache: SLOTS rows of D floats
@group(0) @binding(1) var<storage, read> slots: array<u32>;      // this step's slots, from cache.lookup()
@group(0) @binding(2) var<storage, read_write> out: array<f32>;  // n rows of D floats
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let d = g.x; let i = g.y;
  if (d >= p.D || i >= p.n) { return; }
  out[i * p.D + d] = rows[slots[i] * p.D + d];
}`;

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();

  // 1. The table on disk, written once (the fingerprint changes when the source changes).
  const file = await RowFile.open({ key: 'embedding-demo', rowBytes: ROW_BYTES, rows: V });
  const fp = fingerprint([new TextEncoder().encode(`embedding-demo/1 ${V}x${D}`)]);
  let writeMs = 0;
  if (!file.matches(fp)) {
    log(`writing ${(V * ROW_BYTES / 2 ** 20).toFixed(0)} MB to OPFS…`);
    writeMs = await file.write(fp, (row0, n, dst) => {
      const f = new Float32Array(dst.buffer, dst.byteOffset, n * D);
      for (let r = 0; r < n; r++) for (let d = 0; d < D; d++) f[r * D + d] = value(row0 + r, d);
    });
  }
  await file.openRead();

  // 2. The GPU side: the row cache's buffer, a slot list, the output, the kernel.
  const STORAGE = GPUBufferUsage.STORAGE, DST = GPUBufferUsage.COPY_DST, SRC = GPUBufferUsage.COPY_SRC;
  const rowsBuf = device.createBuffer({ size: SLOTS * ROW_BYTES, usage: STORAGE | DST });
  const slotBuf = device.createBuffer({ size: BATCH * 4, usage: STORAGE | DST });
  const outBuf = device.createBuffer({ size: BATCH * ROW_BYTES, usage: STORAGE | SRC });
  const readBuf = device.createBuffer({ size: BATCH * ROW_BYTES, usage: GPUBufferUsage.MAP_READ | DST });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([D, BATCH, 0, 0]));
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: GATHER }), entryPoint: 'main' } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [rowsBuf, slotBuf, outBuf, params].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const cache = new RowCache({ file, slots: SLOTS, planes: [{ offset: 0, bytes: ROW_BYTES, buffer: rowsBuf }], queue: device.queue });

  // 3. The steps. Token ids are skewed toward low ids, as real token frequencies are.
  let seed = 11; const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  let mismatches = 0, checked = 0; const t0 = performance.now();
  for (let step = 0; step < STEPS; step++) {
    const ids = Array.from({ length: BATCH }, () => Math.floor(V * rnd() ** 8));
    const slots = cache.lookup(ids);                          // misses are read from disk and uploaded here
    device.queue.writeBuffer(slotBuf, 0, slots);
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(D / 64), BATCH);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, BATCH * ROW_BYTES);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    for (let i = 0; i < BATCH; i++) for (let d = 0; d < D; d++) { checked++; if (got[i * D + d] !== value(ids[i], d)) mismatches++; }
  }
  const stepMs = (performance.now() - t0) / STEPS;
  file.close();
  const s = cache.stats;
  self.postMessage({
    ok: mismatches === 0,
    onDisk: { rows: V, bytes: V * ROW_BYTES, writeMs: Math.round(writeMs) },
    onGpu: { rows: SLOTS, bytes: SLOTS * ROW_BYTES },
    steps: STEPS, ids: s.ids, misses: s.misses, hitRate: +(1 - s.misses / s.ids).toFixed(3),
    msPerStep: +stepMs.toFixed(2), valuesChecked: checked, mismatches,
  });
} catch (e) {
  self.postMessage({ ok: false, error: String((e && e.message) || e) });
}
