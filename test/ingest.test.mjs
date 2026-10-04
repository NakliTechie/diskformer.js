// ingest.js pieces that run outside a browser: progress mapping and store keys. (The ingest itself needs OPFS; it
// runs in LocalMind's engines and in examples/.)   node test/ingest.test.mjs
import assert from 'node:assert/strict';
import { ingestProgress, storeKey } from '../src/ingest.js';

const p = ingestProgress({ status: 'ingest', loaded: 5, total: 9, written: 5, secs: 1 });
assert.equal(p.status, 'weights'); assert.equal(p.loaded, 5); assert.equal(p.total, 9);
assert.deepEqual(ingestProgress({ status: 'init' }), { status: 'init' });
assert.equal(storeKey('Qwen3.6-35B-A3B-Q8_0.gguf'), 'qwen3.6-35b-a3b-q8_0');
assert.equal(storeKey('gemma-4-26B_q4_0-it.gguf'), 'gemma-4-26b_q4_0-it');
console.log('ingest: ok (progress mapping, store keys)');
