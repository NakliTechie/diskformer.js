// The rows demo's worker: RowFile needs a dedicated worker (OPFS sync access handles).
import { RowFile, RowCache, fingerprint, measure } from '../../index.js';

const ROWS = 65536, RB = 1024, SLOTS = 2048;
const byteOf = (row, j) => (row * 131 + j * 7 + (row >>> 8)) & 255;
const log = (s) => self.postMessage({ log: s });
try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  const fp = fingerprint([new TextEncoder().encode(`rows-demo/1 ${ROWS}x${RB}`)]);
  const file = await RowFile.open({ key: 'rows-demo', rowBytes: RB, rows: ROWS });
  let writeMs = 0;
  if (!file.matches(fp)) {
    log('writing the table to OPFS…');
    writeMs = await file.write(fp, (row0, n, dst) => { for (let r = 0; r < n; r++) for (let j = 0; j < RB; j++) dst[r * RB + j] = byteOf(row0 + r, j); });
  }
  await file.openRead();
  const plane = device.createBuffer({ size: SLOTS * RB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const map = device.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const cache = new RowCache({ file, slots: SLOTS, planes: [{ offset: 0, bytes: RB, buffer: plane }], queue: device.queue, mapBuffer: map });
  const warmMs = cache.warm(0, 512);
  // 500 lookups of 16 ids, skewed toward low ids the way token frequencies are (a few hundred rows carry most).
  let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  const t0 = performance.now();
  for (let i = 0; i < 500; i++) cache.lookup(Array.from({ length: 16 }, () => Math.floor(ROWS * rnd() ** 12)));
  const lookupMs = (performance.now() - t0) / 500;
  // Read the GPU back: every resident row's bytes, and its map entry.
  const rbP = device.createBuffer({ size: SLOTS * RB, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const rbM = device.createBuffer({ size: ROWS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(plane, 0, rbP, 0, SLOTS * RB); enc.copyBufferToBuffer(map, 0, rbM, 0, ROWS * 4);
  device.queue.submit([enc.finish()]);
  await Promise.all([rbP.mapAsync(GPUMapMode.READ), rbM.mapAsync(GPUMapMode.READ)]);
  const gpu = new Uint8Array(rbP.getMappedRange()), gmap = new Uint32Array(rbM.getMappedRange());
  let resident = 0, rowMismatches = 0, mapMismatches = 0;
  for (let id = 0; id < ROWS; id++) {
    const s = cache.slotOf[id];
    if (gmap[id] !== (s < 0 ? 0xffffffff : s)) mapMismatches++;
    if (s < 0) continue;
    resident++;
    for (let j = 0; j < RB; j++) if (gpu[s * RB + j] !== byteOf(id, j)) { rowMismatches++; break; }
  }
  file.close();
  const disk = await measure({ path: 'diskformer/rows-demo/rows.bin', recordBytes: 64 << 10, reads: 64 });
  const s = cache.stats;
  self.postMessage({
    ok: rowMismatches === 0 && mapMismatches === 0 && resident > 0,
    table: { rows: ROWS, rowBytes: RB, writeMs: Math.round(writeMs) }, slots: SLOTS, warmMs: +warmMs.toFixed(1),
    lookups: s.lookups, ids: s.ids, misses: s.misses, hitRate: +(1 - s.misses / s.ids).toFixed(3), msPerLookup: +lookupMs.toFixed(3),
    resident, rowMismatches, mapMismatches, disk,
  });
} catch (e) {
  self.postMessage({ ok: false, error: String(e && e.message || e) });
}
