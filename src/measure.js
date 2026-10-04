/* measure.js — how fast this machine reads an OPFS file the way a disk tier does. Run it where the tier will run
 * (a page or a worker); OPFS speed varies a lot between machines and browsers.
 *
 *   const r = await measure();                         // writes a 256 MB probe, measures, removes it
 *   const r = await measure({ path: 'diskformer/<key>/experts.bin', recordBytes: 3345408 });   // an existing store
 *   // → { bytes, sequentialMBps, random: { recordBytes, reads, p50Ms, p90Ms, MBps }, burst8: { p50Ms, MBps } }
 *
 * A probe the call just wrote is likely in the OS page cache, so its numbers are an upper bound; point `path` at a
 * store larger than RAM to see the disk itself.
 */
import { OpfsReaderPool, OpfsWriter, removeOpfs } from './opfs-reader.js';

const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

export async function measure({ path = null, bytes = 256 << 20, recordBytes = 3 << 20, reads = 64, workers = 4, chunkBytes = 64 << 20 } = {}) {
  const probe = !path;
  if (probe) {
    path = 'diskformer/.measure/probe.bin';
    const w = await OpfsWriter.open(path, { truncate: true });
    const chunk = new Uint8Array(Math.min(chunkBytes, bytes));
    for (let i = 0; i < chunk.length; i += 4096) chunk[i] = i & 255;
    for (let at = 0; at < bytes; at += chunk.length) await w.write(chunk.subarray(0, Math.min(chunk.length, bytes - at)), at);
    await w.close();
  }
  const pool = await OpfsReaderPool.open(path, { workers });
  try {
    const size = pool.size;
    // Sequential: the load pattern (64 MB chunks, one at a time per worker).
    let t0 = performance.now(), done = 0, inflight = [];
    for (let at = 0; at < size; at += chunkBytes) {
      inflight.push(pool.read(at, Math.min(chunkBytes, size - at)).then((r) => { done += r.got; }));
      if (inflight.length >= workers) { await Promise.all(inflight); inflight = []; }
    }
    await Promise.all(inflight);
    const sequentialMBps = done / 1e6 / ((performance.now() - t0) / 1000);
    // Random whole records: the per-token pattern, one at a time, then bursts of 8 (one MoE layer's misses).
    const n = Math.max(1, Math.floor(size / recordBytes));
    const at = () => Math.floor(Math.random() * n) * recordBytes;
    // Throughput comes from the total time: browsers coarsen performance.now() (0.1 ms in a worker without
    // cross-origin isolation), which rounds a single small read to zero.
    const one = []; let t1 = performance.now();
    for (let i = 0; i < reads; i++) { const s = performance.now(); await pool.read(at(), recordBytes); one.push(performance.now() - s); }
    const oneMs = performance.now() - t1;
    const burst = [], bursts = Math.max(1, Math.round(reads / 8)); t1 = performance.now();
    for (let i = 0; i < bursts; i++) {
      const s = performance.now();
      await Promise.all(Array.from({ length: 8 }, () => pool.read(at(), recordBytes)));
      burst.push(performance.now() - s);
    }
    const burstMs = performance.now() - t1;
    const mbps = (count, ms) => Math.round((count * recordBytes) / 1e6 / (Math.max(ms, 0.1) / 1000));
    return {
      path, bytes: size, workers, sequentialMBps: Math.round(sequentialMBps),
      random: { recordBytes, reads, p50Ms: +pct(one, 0.5).toFixed(2), p90Ms: +pct(one, 0.9).toFixed(2), MBps: mbps(reads, oneMs) },
      burst8: { bursts, p50Ms: +pct(burst, 0.5).toFixed(2), MBps: mbps(8 * bursts, burstMs) },
    };
  } finally {
    await pool.close();
    if (probe) await removeOpfs('diskformer/.measure').catch(() => {});
  }
}
