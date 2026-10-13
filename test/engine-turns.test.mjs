// The engines' turn lock, dispose and error codes, without a GPU: generateTurn is stubbed, so this checks only the
// wrapper in engines/qwen3_moe_ssd.js (every engine inherits it).   node test/engine-turns.test.mjs
import assert from 'node:assert/strict';
import { Qwen3MoeSsd } from '../engines/qwen3_moe_ssd.js';

const tick = () => new Promise((r) => setTimeout(r, 1));
// An engine whose turn appends to a shared log, one entry per "step", with a tick between steps.
function engine(log, { failAt = -1 } = {}) {
  const e = Object.create(Qwen3MoeSsd.prototype);
  e.maxCtx = 8;
  e.generateTurn = async function* (name, { maxNewTokens = 3 } = {}) {
    for (let i = 0; i < maxNewTokens; i++) {
      await tick();
      if (i === failAt) throw new Error('GPU validation error');
      log.push(`${name}${i}`);
      yield { text: `${name}${i}`, tokens: i + 1 };
    }
  };
  e.greedyTurn = async () => { log.push('greedy'); return { ids: [1] }; };
  return e;
}
const drain = async (it) => { const out = []; for await (const r of it) out.push(r.text); return out; };

// Two overlapping generate() calls run one after the other, never interleaved; greedy() waits its turn too.
{
  const log = [], e = engine(log);
  await Promise.all([drain(e.generate('a')), drain(e.generate('b')), e.greedy([1], 1)]);
  assert.deepEqual(log, ['a0', 'a1', 'a2', 'b0', 'b1', 'b2', 'greedy']);
}
// Breaking out early releases the turn.
{
  const log = [], e = engine(log);
  for await (const r of e.generate('a')) if (r.tokens === 1) break;
  assert.deepEqual(await drain(e.generate('b')), ['b0', 'b1', 'b2']);
}
// A failing step surfaces as EXECUTION_FAILED with the original as cause, and releases the turn.
{
  const log = [], e = engine(log, { failAt: 1 });
  await assert.rejects(drain(e.generate('a')), (err) => err.code === 'EXECUTION_FAILED' && err.cause.message === 'GPU validation error');
  e.generateTurn = engine([]).generateTurn;
  assert.equal((await drain(e.generate('b'))).length, 3);
}
// dispose() during a generate: it waits for the step in flight, the generate ends with RESOURCE_DISPOSED, and any
// later or queued call is refused with the same code.
{
  const log = [], e = engine(log);
  e.xs = { drain: async () => {} }; e.expertReader = { close: async () => {} };
  let destroyedAfter = null; e.device = { destroy: () => { destroyedAfter = [...log]; } };
  const it = e.generate('a', { maxNewTokens: 5 });
  const first = it.next();                                     // a generator takes its turn on its first next()
  const queued = drain(e.generate('q'));
  assert.equal((await first).value.text, 'a0');
  const pending = it.next();                                   // step a1 in flight
  await e.dispose();
  assert.deepEqual(destroyedAfter, ['a0', 'a1'], 'the device outlives the step in flight');
  assert.equal((await pending).value.text, 'a1');
  await assert.rejects(it.next(), (err) => err.code === 'RESOURCE_DISPOSED');
  await assert.rejects(queued, (err) => err.code === 'RESOURCE_DISPOSED');
  await assert.rejects(drain(e.generate('c')), (err) => err.code === 'RESOURCE_DISPOSED');
}
// Context full is coded.
{
  const e = engine([]);
  assert.equal(e.contextFull().code, 'CONTEXT_FULL'); assert.equal(e.contextFull().detail.maxCtx, 8);
}
console.log('engine-turns: ok (generate/greedy serialized; early break and failures release the turn; dispose waits for the step in flight; RESOURCE_DISPOSED / EXECUTION_FAILED / CONTEXT_FULL)');
