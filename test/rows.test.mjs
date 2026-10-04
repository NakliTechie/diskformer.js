// RowCache (src/rows.js) with a fake RowFile and a fake GPUQueue: residency, the LRU order, plane uploads, the
// id→slot map writes, the one-lookup capacity check, and warm().   node test/rows.test.mjs
import assert from 'node:assert/strict';
import { RowCache, fingerprint } from '../src/rows.js';

const ROWS = 64, RB = 8;
const file = { rows: ROWS, rowBytes: RB, reads: 0, readRows(row, count, dst) { this.reads++; for (let i = 0; i < count * RB; i++) dst[i] = (row * RB + i) & 255; } };
const writes = [];
const queue = { writeBuffer: (buffer, offset, data, dataOffset = 0, size) => writes.push({ buffer, offset, bytes: new Uint8Array(data.buffer || data, (data.byteOffset || 0) + dataOffset * (data.BYTES_PER_ELEMENT || 1), size ?? (data.byteLength - dataOffset)).slice() }) };
const planes = [{ offset: 0, bytes: 5, buffer: 'A' }, { offset: 5, bytes: 3, buffer: 'B' }];

{ // misses install rows into LRU slots; planes get the row's byte ranges; hits do not read
  const c = new RowCache({ file, slots: 4, planes, queue, mapBuffer: 'M' });
  writes.length = 0;
  const s = c.lookup([7, 9]);
  assert.deepEqual([...s], [0, 1]);
  assert.equal(c.stats.misses, 2);
  const a = writes.find((w) => w.buffer === 'B' && w.offset === 1 * 3);
  assert.deepEqual([...a.bytes], [9 * RB + 5, 9 * RB + 6, 9 * RB + 7]);
  assert.ok(writes.some((w) => w.buffer === 'M' && w.offset === 9 * 4), 'map entry for id 9');
  const reads = file.reads;
  assert.deepEqual([...c.lookup([9, 7])], [1, 0]);
  assert.equal(file.reads, reads, 'hits read nothing');
}
{ // LRU: the least recently used row is evicted, and its map entry is cleared
  const c = new RowCache({ file, slots: 3, planes, queue, mapBuffer: 'M' });
  c.lookup([1]); c.lookup([2]); c.lookup([3]); c.lookup([1]);   // LRU order 2, 3, 1
  writes.length = 0;
  c.lookup([4]);                                                // evicts 2
  assert.ok(!c.has(2) && c.has(1) && c.has(3) && c.has(4));
  const cleared = writes.find((w) => w.buffer === 'M' && w.offset === 2 * 4);
  assert.deepEqual([...new Uint32Array(cleared.bytes.buffer)], [0xffffffff]);
}
{ // one lookup may not need more rows than there are slots
  const c = new RowCache({ file, slots: 2, planes, queue });
  assert.throws(() => c.lookup([1, 2, 3]), /more than 2 rows/);
}
{ // warm(): rows land in contiguous runs and the whole map is written once
  const c = new RowCache({ file, slots: 8, planes, queue, mapBuffer: 'M' });
  writes.length = 0;
  c.warm(10, 6, 4);
  for (let id = 10; id < 16; id++) assert.ok(c.has(id));
  assert.equal(writes.filter((w) => w.buffer === 'M').length, 1);
  assert.equal(c.lookup([12])[0], c.slotOf[12]);
}
// fingerprint: stable for equal bytes, different when a sampled byte changes
{
  const u = new Uint8Array(200000).map((_, i) => i & 255), v = u.slice();
  assert.equal(fingerprint([u]), fingerprint([v]));
  v[0] ^= 1;
  assert.notEqual(fingerprint([u]), fingerprint([v]));
}
console.log('rows: ok (RowCache residency, LRU, planes, map, warm; fingerprint)');
