// ingestGguf resume and cleanup, with fake OPFS files and a fake range server: an abort, an abort during the resume,
// a changed source (same header, new ETag), and failures while opening or writing must leave no file open, and every
// completed store must equal a clean ingest byte for byte.   node test/ingest-resume.test.mjs
import assert from 'node:assert/strict';
import { ingestGguf, fileFetch } from '../src/ingest.js';
import { GGML } from '../src/gguf.js';

// A GGUF v3 header with one F32 tensor, then `n` data bytes from `fill`.
function gguf(n, fill) {
  const out = []; const u8 = (v) => out.push(v & 255);
  const u32 = (v) => { for (let i = 0; i < 4; i++) u8(v >>> (8 * i)); };
  const u64 = (v) => { u32(v); u32(0); };
  const str = (s) => { const b = new TextEncoder().encode(s); u64(b.length); b.forEach(u8); };
  'GGUF'.split('').forEach((c) => u8(c.charCodeAt(0))); u32(3); u64(1); u64(1);
  str('general.architecture'); u32(8); str('demo');
  str('w'); u32(1); u64(n / 4); u32(GGML.F32); u64(0);
  while (out.length % 32) u8(0);
  const head = out.length, file = new Uint8Array(head + n);
  file.set(out);
  for (let i = 0; i < n; i++) file[head + i] = fill(i);
  return { file, head };
}

// 40 units of 100 bytes, alternating between the two output files; ranges of 256 bytes, so units straddle ranges.
const N = 4000, UNIT = 100;
const plan = (head) => ({
  layout: () => ({ dense: { file: 'dense.bin', bytes: N / 2 }, experts: { file: 'experts.bin', bytes: N / 2 } }),
  units: () => Array.from({ length: N / UNIT }, (_, i) => ({ src: head + i * UNIT, len: UNIT, file: i % 2 ? 'experts' : 'dense', raw: (i >> 1) * UNIT })),
});

// Fake OPFS: files as growable byte arrays; counts open writers; can fail an open or a write at a given offset.
function fakeIo() {
  const files = new Map(), fail = { open: null, writeAt: null };
  let open = 0;
  const OpfsWriter = {
    async open(path, { truncate = false } = {}) {
      if (fail.open && path.endsWith(fail.open)) throw new Error(`cannot open ${path}`);
      if (truncate || !files.has(path)) files.set(path, new Uint8Array(0));
      open++;
      const put = (src, at) => {
        let f = files.get(path);
        if (f.length < at + src.length) { const g = new Uint8Array(at + src.length); g.set(f); f = g; files.set(path, f); }
        f.set(src, at);
      };
      return {
        async write(u8, at) { put(u8, at); },
        async writeOwned(buf, len, at) {
          await new Promise((r) => setTimeout(r, 0));
          if (fail.writeAt !== null && path.endsWith('experts.bin') && at === fail.writeAt) throw new Error('write failed');
          put(new Uint8Array(buf, 0, len), at); return buf;
        },
        async flush() {},
        async close() { open--; },
      };
    },
  };
  const text = new Map();
  return { files, fail, openCount: () => open, io: { OpfsWriter, readOpfsText: async (p) => text.get(p) ?? null, writeOpfsText: async (p, t) => { text.set(p, t); } } };
}

// Fake range server: 206 responses streamed in 64-byte chunks, honouring the abort signal.
const server = (file, etag) => async (url, { headers, signal }) => {
  const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.Range).map(Number);
  const end = Math.min(b, file.length - 1);
  let at = a;
  return {
    ok: true, status: 206,
    headers: { get: (k) => (k === 'etag' ? etag : k === 'content-range' ? `bytes ${a}-${end}/${file.length}` : null) },
    arrayBuffer: async () => file.slice(a, end + 1).buffer,
    body: { getReader: () => ({ async read() {
      if (signal) signal.throwIfAborted();
      await new Promise((r) => setTimeout(r, 0));
      if (at > end) return { done: true };
      const v = file.slice(at, Math.min(end + 1, at + 64)); at += v.length; return { done: false, value: v };
    } }) },
  };
};

async function run(fs, src, { abortAt = Infinity, etag = 'e1' } = {}) {
  const ctrl = new AbortController(), events = [];
  try {
    const m = await ingestGguf({ url: 'm.gguf', key: 'k', plan: plan(src.head), fetch: server(src.file, etag), signal: ctrl.signal, rangeBytes: 256, io: fs.io,
      onProgress: (e) => { events.push(e); if (e.status === 'ingest' && e.loaded / e.total >= abortAt) ctrl.abort(); } });
    return { m, events };
  } catch (error) { return { error, events }; }
}
const bytes = (fs) => [fs.files.get('diskformer/k/dense.bin'), fs.files.get('diskformer/k/experts.bin')].map((f) => Array.from(f));

const a = gguf(N, (i) => (i * 7) & 255), b = gguf(N, (i) => (i * 13 + 5) & 255);
const cleanA = fakeIo(); assert.ok((await run(cleanA, a)).m.complete); const refA = bytes(cleanA);
const cleanB = fakeIo(); assert.ok((await run(cleanB, b)).m.complete); const refB = bytes(cleanB);

// Abort, abort again during the resume, then finish: the result equals the clean ingest.
{
  const fs = fakeIo();
  const r1 = await run(fs, a, { abortAt: 0.55 });
  assert.equal(r1.error.name, 'AbortError'); assert.equal(fs.openCount(), 0);
  const r2 = await run(fs, a, { abortAt: 0.85 });
  assert.equal(r2.error.name, 'AbortError'); assert.equal(fs.openCount(), 0);
  const resumed2 = r2.events.find((e) => e.status === 'ingest-resume');
  assert.ok(resumed2 && resumed2.unit > 0, 'the second run resumes');
  const r3 = await run(fs, a);
  const resumed3 = r3.events.find((e) => e.status === 'ingest-resume');
  assert.ok(resumed3 && resumed3.unit > resumed2.unit, 'the third run resumes past the second');
  assert.equal(r3.m.resumedAtUnit, resumed3.unit);
  const firstIngest = r3.events.find((e) => e.status === 'ingest');
  assert.equal(firstIngest.resumedFrom, resumed3.loaded); assert.ok(firstIngest.loaded > firstIngest.resumedFrom);
  assert.deepEqual(bytes(fs), refA); assert.equal(fs.openCount(), 0);
}
// Same header, new ETag (the file changed): start over, never mix the two.
{
  const fs = fakeIo();
  assert.equal((await run(fs, a, { abortAt: 0.6 })).error.name, 'AbortError');
  const r = await run(fs, b, { etag: 'e2' });
  assert.ok(!r.events.some((e) => e.status === 'ingest-resume'), 'a changed source does not resume');
  assert.deepEqual(bytes(fs), refB);
}
// A failed open and a failed final write leave nothing open.
{
  const fs = fakeIo(); fs.fail.open = 'experts.bin';
  assert.match((await run(fs, a)).error.message, /cannot open/); assert.equal(fs.openCount(), 0);
}
{
  const fs = fakeIo(); fs.fail.writeAt = N / 2 - UNIT;          // the last unit of experts.bin
  assert.match((await run(fs, a)).error.message, /write failed/); assert.equal(fs.openCount(), 0);
}
// A local file as the source (fileFetch over a Blob): the same store as the download, byte for byte; an abort
// resumes from the same file; a download's interrupted state is not resumed from a file (no ETag to match).
{
  const runFile = (fs, src, { abortAt = Infinity } = {}) => {
    const ctrl = new AbortController(), events = [];
    return ingestGguf({ url: 'm.gguf', key: 'k', plan: plan(src.head), fetch: fileFetch(new Blob([src.file])), signal: ctrl.signal, rangeBytes: 256, io: fs.io,
      onProgress: (e) => { events.push(e); if (e.status === 'ingest' && e.loaded / e.total >= abortAt) ctrl.abort(); } })
      .then((m) => ({ m, events }), (error) => ({ error, events }));
  };
  const fs = fakeIo();
  assert.ok((await runFile(fs, a)).m.complete); assert.deepEqual(bytes(fs), refA); assert.equal(fs.openCount(), 0);
  const fs2 = fakeIo();
  assert.equal((await runFile(fs2, a, { abortAt: 0.5 })).error.name, 'AbortError'); assert.equal(fs2.openCount(), 0);
  const r = await runFile(fs2, a);
  assert.ok(r.events.some((e) => e.status === 'ingest-resume'), 'a file ingest resumes from the same file');
  assert.deepEqual(bytes(fs2), refA);
  const fs3 = fakeIo();
  assert.equal((await run(fs3, a, { abortAt: 0.5 })).error.name, 'AbortError');
  const r3 = await runFile(fs3, a);
  assert.ok(!r3.events.some((e) => e.status === 'ingest-resume'), 'a download is not resumed from a file');
  assert.deepEqual(bytes(fs3), refA);
  const ff = fileFetch(new Blob([new Uint8Array(10)]));
  const r206 = await ff('x', { headers: { Range: 'bytes=4-99' } });
  assert.equal(r206.status, 206); assert.equal(r206.headers.get('content-range'), 'bytes 4-9/10'); assert.equal((await r206.arrayBuffer()).byteLength, 6);
  assert.equal((await ff('x', { headers: { Range: 'bytes=10-19' } })).status, 416);
}
console.log('ingest-resume: ok (abort + abort during resume + finish = clean bytes; changed source starts over; failures close every file; a local file ingests the same bytes and resumes from itself)');
