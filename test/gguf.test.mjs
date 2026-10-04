// gguf.js: parseGguf on a synthetic header (every value type the loader meets), tensorBytes, and splitQ8 / splitQ4
// bit-exactness against ggml's dequantization.   node test/gguf.test.mjs
import assert from 'node:assert/strict';
import { parseGguf, tensorBytes, splitQ8, splitQ4, GGML, Q8_BLOCK, Q4_BLOCK } from '../src/gguf.js';

const f16 = (h) => { const s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15); };
let x = 99; const rnd = () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) & 255; };

// A GGUF v3 header: string, u32, f32, bool, string array, u32 array; two tensors; 32-byte alignment.
{
  const out = []; const u8 = (n) => out.push(n & 255);
  const u32 = (n) => { for (let i = 0; i < 4; i++) u8(n >>> (8 * i)); };
  const u64 = (n) => { u32(n); u32(0); };
  const str = (s) => { const b = new TextEncoder().encode(s); u64(b.length); b.forEach(u8); };
  const f32 = (v) => { const b = new Uint8Array(new Float32Array([v]).buffer); b.forEach(u8); };
  'GGUF'.split('').forEach((c) => u8(c.charCodeAt(0))); u32(3); u64(2); u64(6);
  str('general.architecture'); u32(8); str('demo');
  str('demo.block_count'); u32(4); u32(4);
  str('demo.rope.freq_base'); u32(6); f32(10000);
  str('demo.flag'); u32(7); u8(1);
  str('tokenizer.ggml.tokens'); u32(9); u32(8); u64(2); str('a'); str('b');
  str('demo.pattern'); u32(9); u32(4); u64(3); u32(1); u32(0); u32(1);
  str('blk.0.w'); u32(2); u64(64); u64(2); u32(GGML.Q8_0); u64(0);
  str('blk.0.n'); u32(1); u64(64); u32(GGML.F32); u64(128);
  const h = parseGguf(Uint8Array.from(out));
  assert.equal(h.version, 3);
  assert.equal(h.kv['general.architecture'], 'demo');
  assert.equal(h.kv['demo.block_count'], 4);
  assert.equal(h.kv['demo.rope.freq_base'], 10000);
  assert.equal(h.kv['demo.flag'], true);
  assert.deepEqual(h.kv['tokenizer.ggml.tokens'], ['a', 'b']);
  assert.deepEqual(h.kv['demo.pattern'], [1, 0, 1]);
  assert.equal(h.tensors.length, 2);
  assert.equal(tensorBytes(h.tensors[0]), (128 / 32) * Q8_BLOCK);
  assert.equal(tensorBytes(h.tensors[1]), 256);
  assert.equal(h.dataStart % 32, 0);
  assert.throws(() => parseGguf(Uint8Array.from(out.slice(0, 40))), (e) => e.needBytes > 40);
}
// splitQ8: value plane + f16-scale plane reproduce every dequantized value exactly.
{
  const nb = 37, src = new Uint8Array(nb * Q8_BLOCK);
  for (let i = 0; i < src.length; i++) src[i] = rnd();
  for (let b = 0; b < nb; b++) src[b * Q8_BLOCK + 1] &= 0x3b;
  const q = new Uint8Array(nb * 32), s = new Uint8Array(nb * 2);
  splitQ8(src, q, s);
  for (let b = 0; b < nb; b++) for (let j = 0; j < 32; j++) {
    const d = f16(src[b * Q8_BLOCK] | (src[b * Q8_BLOCK + 1] << 8)), d2 = f16(s[2 * b] | (s[2 * b + 1] << 8));
    const v = src[b * Q8_BLOCK + 2 + j], v2 = q[b * 32 + j];
    assert.equal((v > 127 ? v - 256 : v) * d, (v2 > 127 ? v2 - 256 : v2) * d2);
  }
}
// splitQ4: nibble plane (ggml order: low nibbles are values 0..15) + f16-scale plane, exact.
{
  const nb = 41, src = new Uint8Array(nb * Q4_BLOCK);
  for (let i = 0; i < src.length; i++) src[i] = rnd();
  for (let b = 0; b < nb; b++) src[b * Q4_BLOCK + 1] &= 0x3b;
  const q = new Uint8Array(nb * 16), s = new Uint8Array(nb * 2);
  splitQ4(src, q, s);
  for (let b = 0; b < nb; b++) {
    const d = f16(src[b * Q4_BLOCK] | (src[b * Q4_BLOCK + 1] << 8)), d2 = f16(s[2 * b] | (s[2 * b + 1] << 8));
    for (let j = 0; j < 16; j++) {
      const v = src[b * Q4_BLOCK + 2 + j], v2 = q[b * 16 + j];
      assert.equal(((v & 15) - 8) * d, ((v2 & 15) - 8) * d2);
      assert.equal(((v >> 4) - 8) * d, ((v2 >> 4) - 8) * d2);
    }
  }
}
console.log('gguf: ok (header parse, tensorBytes, splitQ8, splitQ4)');
