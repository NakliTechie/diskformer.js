// denoise.js: the block-diffusion decoding loop, with no runtime inside.
//
// The caller supplies one async forward pass:
//
//   forward({ mode, start, ids, wantLogits }) -> Float32Array | null
//
//   mode 'preview'  the denoise pass over the current block: reads the committed cache, writes
//                   nothing, returns [ids.length * V] fp32 logits (row-major). wantLogits = 'all'.
//   mode 'commit'   the causal pass: appends ids at absolute positions [start, start + ids.length).
//                   Prefill is the first commit; commits arrive in order with no gaps.
//                   wantLogits 'last' returns the last position's V logits; 'none' returns null.
//   Optional fast path (option top1): a greedy, penalty-free preview asks wantLogits 'top1' and gets
//   { token: Int32Array(n), conf: Float32Array(n) }: the per-row argmax with the mask id excluded and
//   its softmax probability at T = 1 (mask logit at -inf). A runtime without it throws, and the loop
//   falls back to 'all' for the rest of the run.
//
// Logits are same-position: row i scores slot start + i (not the next token).
//
// `ids` is an Int32Array; masked slots hold `maskId`. The loop reads the returned array during the
// step and never keeps it, so the caller owns buffers and disposal.
// kohra's ONNX path (kohra.js) and diskformer's clover_moe engine both drive this loop.

const DEFAULTS = {
  maskId: 151669,
  eosIds: [],             // stop tokens; the response is trimmed before the first one
  blockSize: 32,
  maxNewTokens: 128,
  threshold: null,        // number: fill every masked position past it per preview (min 1).
                          // null: fixed reveal schedule over `steps` previews.
  strict: false,          // threshold compare: true uses conf > threshold, false conf >= threshold
  steps: 128,             // fixed-schedule mode only: total previews, split evenly over blocks
  maxStepsPerBlock: null, // threshold mode: throw if a block still has masks after this many previews
  temperature: 0,         // 0 = argmax; > 0 = sample softmax(logits / temperature)
  topK: 0,                // sampling filters (temperature > 0 only); 0 / 1 = off
  topP: 1,
  remasking: 'low_confidence', // or 'random'
  suppressMask: false,    // set the mask token's logit to -inf before scoring
  seed: false,            // a commit's last logit picks the next block's first token
  alignToGrid: false,     // blocks on the absolute pos // blockSize grid (first block holds the prompt tail)
  fullLastBlock: false,   // denoise the last block at full blockSize, then trim to maxNewTokens
  stopAtEos: false,       // stop after a resolved block that holds a stop token
  penalties: null,        // see applyPenalties
  top1: false,            // try the 'top1' preview fast path when greedy and penalty-free
  onStep: null,           // ({ canvas, P, start, end, fresh, block, numBlocks, previews, elapsedMs })
  onBlock: null,          // ({ tokens, block }) after each resolved block (response tokens only)
  yieldEvery: 1,          // release the event loop every N previews
};

// Clover-35B-A3B-Preview: generation_config.json plus the denoising config in modeling_clover_moe.py.
export const PRESETS = {
  clover: {
    maskId: 151669,
    eosIds: [248046, 248044],
    blockSize: 32,
    threshold: 0.95,
    strict: true,
    maxStepsPerBlock: 32,
    suppressMask: true,
    fullLastBlock: true,
    stopAtEos: true,
    penalties: {
      presence: 0.9, frequency: 0, repetition: 1.05,
      length: 10, lengthStart: 8192, lengthWindow: 8192,
      thinkEnd: 10, thinkEndStart: 4096, thinkEndWindow: 4096, thinkEndId: 248069,
      stopIds: [248044, 248046],   // eos ids plus pad (248044)
    },
  },
};

// Cooperative yield that lets the UI paint without waiting on the compositor.
// requestAnimationFrame stalls in a hidden tab; a MessageChannel macrotask always fires.
export const yieldToEventLoop = (() => {
  if (typeof MessageChannel === 'undefined') {
    return () => new Promise((resolve) => setTimeout(resolve, 0));
  }
  const ch = new MessageChannel();
  const queue = [];
  // ref/unref exist only in Node: hold the port open while a yield is pending, so an
  // idle port never keeps the process alive and a pending one never lets it exit.
  ch.port1.onmessage = () => { queue.shift()?.(); if (!queue.length) ch.port1.unref?.(); };
  ch.port1.unref?.();
  return () => new Promise((resolve) => { queue.push(resolve); ch.port1.ref?.(); ch.port2.postMessage(0); });
})();

// Per-block reveal counts for the linear-alpha (MDLM) scheduler: round remaining/(steps-i)
// per step, skip zero entries. Mirrors dllm get_num_transfer_tokens.
export function transferSchedule(nMasked, steps) {
  const ks = [];
  let remaining = nMasked;
  for (let i = 0; i < steps && remaining > 0; i++) {
    const k = Math.min(Math.round(remaining / (steps - i)), remaining);
    if (k > 0) { ks.push(k); remaining -= k; }
  }
  return ks;
}

// Response-only penalties, in CloverPenaltyState.apply order (sampling_clover_moe.py):
// frequency, presence, repetition over committed response tokens, then the length ramp on
// stop ids and the think-end ramp. `counts` holds committed response tokens only; tokens still
// in the block never penalise themselves. Mutates rows of `logits` in place.
export function applyPenalties(logits, rows, V, { counts, length, block, maskId, p }) {
  // Every operation rounds to fp32 the way PyTorch's fp32 kernels do, with scalars cast to fp32.
  const fp = Math.fround;
  const ramp = (row, start, window, strength) =>
    fp(fp(Math.min(Math.max(fp(fp(length + row + 1 - start) / fp(window)), 0), 1)) * fp(strength));
  let closeOpen = false;
  if (p.thinkEnd) {
    closeOpen = !counts.has(p.thinkEndId);
    for (let i = 0; i < block.length && closeOpen; i++) {
      if (block[i] !== maskId && block[i] === p.thinkEndId) closeOpen = false;
    }
  }
  const freq = fp(p.frequency ?? 0), pres = fp(p.presence ?? 0), rep = fp(p.repetition ?? 1);
  for (let r = 0; r < rows; r++) {
    const off = r * V;
    for (const [tok, n] of counts) {
      let l = logits[off + tok];
      if (freq) l = fp(l - fp(n * freq));
      if (pres) l = fp(l - pres);
      if (rep !== 1) l = fp(l < 0 ? l * rep : l / rep);
      logits[off + tok] = l;
    }
    if (p.length && p.stopIds?.length) {
      const add = ramp(r, p.lengthStart, p.lengthWindow, p.length);
      for (const s of p.stopIds) logits[off + s] = fp(logits[off + s] + add);
    }
    if (closeOpen) {
      logits[off + p.thinkEndId] = fp(logits[off + p.thinkEndId] + ramp(r, p.thinkEndStart, p.thinkEndWindow, p.thinkEnd));
    }
  }
}

// Pick a token for one logits row and its confidence (softmax at T = 1 of the same row).
function pick(logits, off, V, cfg) {
  let maxL = -Infinity;
  for (let v = 0; v < V; v++) if (logits[off + v] > maxL) maxL = logits[off + v];
  let sumExp = 0;
  for (let v = 0; v < V; v++) sumExp += Math.exp(logits[off + v] - maxL);
  let best = 0;
  if (cfg.temperature > 0) {
    const keep = samplingFilter(logits, off, V, cfg);
    let bestScore = -Infinity;
    for (let v = 0; v < V; v++) {
      if (keep && !keep(logits[off + v], v)) continue;
      // Gumbel-max on logits / temperature == a draw from softmax(logits / temperature).
      const s = logits[off + v] / cfg.temperature - Math.log(-Math.log(cfg.random() + 1e-12) + 1e-12);
      if (s > bestScore) { bestScore = s; best = v; }
    }
  } else {
    for (let v = 1; v < V; v++) if (logits[off + v] > logits[off + best]) best = v;
  }
  return { tok: best, conf: Math.exp(logits[off + best] - maxL) / sumExp };
}

// top-k then top-p over logits / temperature, as predict_with_confidence does. Returns null when off.
function samplingFilter(logits, off, V, cfg) {
  const k = cfg.topK > 0 && cfg.topK < V ? cfg.topK : 0;
  const topP = cfg.topP != null && cfg.topP < 1 ? cfg.topP : 0;
  if (!k && !topP) return null;
  const order = Array.from({ length: V }, (_, v) => v).sort((a, b) => logits[off + b] - logits[off + a]);
  let n = k || V;
  if (topP) {
    const T = cfg.temperature, top = logits[off + order[0]] / T;
    let z = 0;
    for (let i = 0; i < n; i++) z += Math.exp(logits[off + order[i]] / T - top);
    let cum = 0;
    for (let i = 0; i < n; i++) {
      const pr = Math.exp(logits[off + order[i]] / T - top) / z;
      if (cum > topP) { n = i; break; }       // drop once the mass before this token passes top_p
      cum += pr;
    }
  }
  const allowed = new Set(order.slice(0, Math.max(n, 1)));
  return (_, v) => allowed.has(v);
}

// Run the loop. `promptIds` is an array of token ids. Returns
// { tokenIds, stopToken, canvas, P, previews, commits, seconds }.
export async function denoise({ forward, promptIds, ...options }) {
  const cfg = { random: Math.random, ...DEFAULTS, ...options };
  const MASK = cfg.maskId, bs = cfg.blockSize;
  const eos = new Set(cfg.eosIds);
  const P = promptIds.length;
  const maxNew = cfg.maxNewTokens;

  // Blocks to denoise, left to right. Prompt-aligned: blocks start at P. Grid-aligned:
  // blocks live on pos // blockSize, so the first one holds the prompt tail.
  const first = cfg.alignToGrid ? Math.floor(P / bs) * bs : P;
  const limit = P + maxNew;
  const blocks = [];
  for (let s = first; s < limit; s += bs) blocks.push([s, cfg.fullLastBlock ? s + bs : Math.min(s + bs, limit)]);
  const numBlocks = blocks.length;
  const T = numBlocks ? blocks[numBlocks - 1][1] : P;
  const stepsPerBlock = Math.ceil(cfg.steps / Math.max(numBlocks, 1));

  const canvas = new Int32Array(Math.max(T, P)).fill(MASK);
  canvas.set(promptIds, 0);

  const counts = new Map();      // committed response tokens, for penalties
  let responseLength = 0;
  let previews = 0, commits = 0;
  let top1 = cfg.top1 && !(cfg.temperature > 0) && !cfg.penalties && cfg.remasking !== 'random';
  const t0 = performance.now();

  const commit = async (s, e, next) => {
    const wantLogits = cfg.seed && next ? 'last' : 'none';
    const out = await forward({ mode: 'commit', start: s, ids: canvas.slice(s, e), wantLogits });
    commits++;
    if (wantLogits === 'last' && canvas[next] === MASK) {
      const V = out.length;
      let best = 0;
      for (let v = 1; v < V; v++) if (out[v] > out[best]) best = v;
      canvas[next] = best;
    }
  };

  if (numBlocks && first > 0) await commit(0, first, first);   // prefill

  let stopToken = null, genEnd = limit;
  for (let b = 0; b < numBlocks; b++) {
    const [start, end] = blocks[b];
    const masked = () => { let n = 0; for (let i = start; i < end; i++) if (canvas[i] === MASK) n++; return n; };
    const nMasked = masked();
    const schedule = cfg.threshold == null ? transferSchedule(nMasked, stepsPerBlock) : null;
    let step = 0;

    while (masked() > 0) {
      if (schedule && step >= schedule.length) break;
      if (!schedule && cfg.maxStepsPerBlock != null && step >= cfg.maxStepsPerBlock) {
        throw new Error(`denoise: block ${b} still has masks after ${step} previews`);
      }
      const ids = canvas.slice(start, end);
      const cand = [];
      let fast = null;
      if (top1) {
        try { fast = await forward({ mode: 'preview', start, ids, wantLogits: 'top1' }); } catch { top1 = false; }
      }
      if (fast) {
        previews++;
        for (let p = start; p < end; p++) {
          if (canvas[p] === MASK) cand.push({ p, tok: fast.token[p - start], conf: fast.conf[p - start] });
        }
      } else {
        const logits = await forward({ mode: 'preview', start, ids, wantLogits: 'all' });
        previews++;
        const rows = end - start, V = logits.length / rows;
        if (cfg.suppressMask) for (let r = 0; r < rows; r++) logits[r * V + MASK] = -Infinity;
        if (cfg.penalties) {
          applyPenalties(logits, rows, V, { counts, length: responseLength, block: ids, maskId: MASK, p: cfg.penalties });
        }
        for (let p = start; p < end; p++) {
          if (canvas[p] !== MASK) continue;
          const { tok, conf } = pick(logits, (p - start) * V, V, cfg);
          cand.push({ p, tok, conf: cfg.remasking === 'random' ? cfg.random() : conf });
        }
      }

      let chosen;
      if (schedule) {
        cand.sort((x, y) => y.conf - x.conf);
        chosen = cand.slice(0, schedule[step]);
      } else {
        const thr = cfg.threshold;
        chosen = cand.filter((c) => (cfg.strict ? c.conf > thr : c.conf >= thr));
        if (chosen.length === 0) {               // progress floor: the single most confident
          let top = cand[0];
          for (const c of cand) if (c.conf > top.conf) top = c;
          chosen = [top];
        }
      }
      step++;

      const fresh = new Set();
      for (const { p, tok } of chosen) { canvas[p] = tok; fresh.add(p); }
      cfg.onStep?.({ canvas, P, start, end, fresh, block: b, numBlocks, previews, elapsedMs: performance.now() - t0 });
      if (cfg.yieldEvery && previews % cfg.yieldEvery === 0) await yieldToEventLoop();
    }

    // Response tokens of this block, in position order, cut at the first stop token.
    // Stop only after the whole block resolves: an earlier slot can fill after a later EOS.
    const respStart = Math.max(start, P), respEnd = Math.min(end, limit);
    let cut = respEnd;
    if (cfg.stopAtEos) {
      for (let i = respStart; i < respEnd; i++) if (eos.has(canvas[i])) { cut = i + 1; stopToken = canvas[i]; break; }
    }
    if (cfg.penalties) {
      for (let i = respStart; i < cut; i++) counts.set(canvas[i], (counts.get(canvas[i]) ?? 0) + 1);
    }
    responseLength += cut - respStart;
    cfg.onBlock?.({ tokens: Array.from(canvas.subarray(respStart, stopToken == null ? cut : cut - 1)), block: b });
    if (stopToken != null) { genEnd = cut - 1; break; }
    if (b + 1 < numBlocks && end < limit) await commit(start, end, blocks[b + 1][0]);
  }

  // Trim at the first stop token after the prompt (fixed-schedule runs fill every block first).
  if (stopToken == null) {
    for (let i = P; i < limit; i++) if (eos.has(canvas[i])) { genEnd = i; stopToken = canvas[i]; break; }
  }
  return {
    tokenIds: Array.from(canvas.subarray(P, Math.min(genEnd, limit))),
    stopToken,
    canvas,
    P,
    previews,
    commits,
    seconds: (performance.now() - t0) / 1000,
  };
}
