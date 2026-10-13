// readStore / listStores over fake OPFS: a store counts as a model only when its manifest is complete and every file
// it names is on disk at full size. Zero-byte and short files (crash leftovers) are invalid, never complete.
//   node test/store.test.mjs
import assert from 'node:assert/strict';
import { readStore, listStores, FORMAT } from '../src/ingest.js';
import { DiskformerError, isDiskformerError, codedError } from '../src/errors.js';

const fake = (files) => ({
  readOpfsText: async (p) => (files[p] === undefined ? null : files[p]),
  opfsFileSize: async (p) => (files[p] === undefined ? null : typeof files[p] === 'string' ? files[p].length : files[p]),
  opfsSubdirs: async (root) => [...new Set(Object.keys(files).filter((p) => p.startsWith(`${root}/`)).map((p) => p.split('/')[1]))].sort(),
});
const manifest = (extra = {}) => JSON.stringify({ format: FORMAT, complete: true, source: { dataStart: 100 },
  dense: { file: 'dense.bin', bytes: 1024 }, experts: { file: 'experts.bin', bytes: 4000 }, ...extra });
const store = (key, { m = manifest(), header = 100, dense = 1024 - 200, experts = 4000 } = {}) => ({
  [`diskformer/${key}/manifest.json`]: m, [`diskformer/${key}/header.bin`]: header, [`diskformer/${key}/dense.bin`]: dense, [`diskformer/${key}/experts.bin`]: experts,
});
const state = async (files, key = 'a') => (await readStore(key, { io: fake(files) }));

assert.equal((await state(store('a'))).state, 'complete');                                   // dense within its padding slack
assert.equal((await state({})).state, 'missing');
let r = await state(store('a', { experts: 0 }));
assert.equal(r.state, 'invalid'); assert.match(r.problem, /experts.bin is empty/);
r = await state(store('a', { header: 0 }));
assert.equal(r.state, 'invalid'); assert.match(r.problem, /header.bin is empty/);
r = await state(store('a', { experts: 3999 }));
assert.equal(r.state, 'invalid'); assert.match(r.problem, /experts.bin is 3999 bytes, expected 4000/);
r = await state(store('a', { dense: 1024 - 256 }));
assert.equal(r.state, 'invalid');                                                                // short beyond the slack
assert.equal((await state(store('a', { m: manifest({ experts: { file: 'experts.bin', bytes: 0 } }), experts: 0 }))).state, 'complete');  // dense model
const noExperts = store('a'); delete noExperts['diskformer/a/experts.bin'];
assert.match((await state(noExperts)).problem, /experts.bin is missing/);
assert.equal((await state(store('a', { m: JSON.stringify({ format: FORMAT, complete: false }) }))).state, 'partial');
assert.equal((await state(store('a', { m: '' }))).state, 'invalid');                             // zero-byte manifest
assert.equal((await state(store('a', { m: manifest({ format: 'other/1' }) }))).state, 'invalid');
assert.equal((await state(store('a', { m: '{ not json' }))).problem, 'no readable manifest');

const all = await listStores({ io: fake({ ...store('a'), ...store('b', { experts: 0 }), ...store('c', { m: JSON.stringify({ format: FORMAT, complete: false }) }) }) });
assert.deepEqual(all.map((s) => [s.key, s.state]), [['a', 'complete'], ['b', 'invalid'], ['c', 'partial']]);
assert.equal(all[0].bytes, 100 + 824 + 4000);
const other = await listStores({ io: fake(store('d', { m: manifest({ format: 'engine/1' }) })) });
assert.equal(other[0].state, 'complete');                                                        // any format by default
assert.equal((await listStores({ format: FORMAT, io: fake(store('d', { m: manifest({ format: 'engine/1' }) })) }))[0].state, 'invalid');

// Coded errors: checked by name + code, so a second copy of the module still matches.
const e = new DiskformerError('BUDGET_TOO_SMALL', 'x', { detail: { fixed: 1 } });
assert.ok(isDiskformerError(e, 'BUDGET_TOO_SMALL')); assert.ok(!isDiskformerError(e, 'LOAD_FAILED'));
assert.ok(isDiskformerError({ name: 'DiskformerError', code: 'LOAD_FAILED' }, 'LOAD_FAILED'));
assert.ok(!isDiskformerError(new Error('LOAD_FAILED')));
const ac = new AbortController(); ac.abort();
assert.equal(codedError(new TypeError('network'), { signal: ac.signal }).code, 'LOAD_ABORTED');  // the signal decides
assert.equal(codedError(new Error('HTTP 500')).code, 'LOAD_FAILED');
assert.equal(codedError(e), e);
console.log('store: ok (complete / partial / missing; zero-byte, short and missing files invalid; listStores; coded errors)');
