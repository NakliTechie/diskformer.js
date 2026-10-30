// engines/lfm3_omni.js's pure-JS parts against scripts/lfm3-omni-ref.py's fixture (test/refs/lfm3-omni): log-mel
// features, image preprocessing (resize, normalize, patches), the prompt ids (chat template, placeholders), the
// audio length cap, the Cache Storage layer (fetchCached, against an in-memory stand-in), and the speech-out parts:
// the inverse STFT against numpy (speech-out.json, istft.*), chunked codec decoding, MaskGIT frames, s2s prompt ids.
//   node test/lfm3-omni.test.mjs
// The prompt-id check needs the model's tokenizer files (LFM3_OMNI_DIR, default the Models2 copy) and transformers.js
// 4.3.2 (TRANSFORMERS_JS: its package folder or entry file, else the `@huggingface/transformers` package); without
// them it says so and skips that part.
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { Istft, SpeechStream, sampleFrame, audioSampling, randomOrder, SPEECH_SYSTEM_PROMPT } from '../engines/lfm3_omni.js';
import { Lfm3Omni, fetchCached, checkAudio, MAX_AUDIO_SECONDS, logMel, melFrames, speechTokens, MEL, preprocessImage, imageConfig, expandPlaceholders, sampleToken, smartResize } from '../engines/lfm3_omni.js';

const dir = new URL('./refs/lfm3-omni/', import.meta.url);
const fx = JSON.parse(readFileSync(new URL('fixture.json', dir)));
const bin = (f) => { const b = readFileSync(new URL(f, dir)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const pcm = new Int16Array(bin(fx.speech.file));
const wave = Float32Array.from(pcm, (x) => x / 32768);
let passed = 0;

{ // log-mel: every value within 1e-4 of the reference, relative to its largest magnitude
  const ref = new Float32Array(bin(fx.speech.mel_file));
  const t0 = performance.now();
  const mel = logMel(wave, MEL);
  const ms = performance.now() - t0;
  assert.deepEqual(mel.dims, fx.speech.mel.shape);
  assert.equal(melFrames(wave.length), fx.speech.mel.shape[1]);
  assert.equal(speechTokens(mel.dims[1]), fx.speech.audio_tokens);
  let worst = 0, scale = 0;
  for (let i = 0; i < ref.length; i++) { worst = Math.max(worst, Math.abs(mel.data[i] - ref[i])); scale = Math.max(scale, Math.abs(ref[i])); }
  console.log(`mel [${mel.dims}] in ${ms.toFixed(0)} ms: max |diff| ${worst.toExponential(2)} of scale ${scale.toFixed(3)}`);
  assert.ok(worst <= 1e-4 * scale, `mel max |diff| ${worst} over 1e-4 × ${scale}`);
  passed++;
}
{ // smart_resize: rounding (Python's half-to-even), the pixel floor and ceiling
  assert.deepEqual(smartResize(250, 333, 32, 65536, 2097152), [256, 320]);
  assert.deepEqual(smartResize(48, 80, 32, 65536, 2097152), [224, 352]);
  assert.deepEqual(smartResize(4000, 3000, 32, 65536, 2097152), [1664, 1248]);
  passed++;
}
{ // image: grid, the checksums of pixel_values and its sampled values. The resize is integer arithmetic (torch's
  // uint8 kernel), so the values are exact; the bounds below leave one uint8 step (2/255) for a rounding flip.
  const { width, height, grid } = fx.image, ref = fx.image.pixel_values;
  const rgb = new Uint8Array(bin(fx.image.file));
  // imageConfig()'s defaults are LFM3-Omni-2B's preprocessor_config.json
  const p = preprocessImage({ data: rgb, width, height, channels: 3 }, imageConfig());
  assert.deepEqual(p.grid, grid);
  assert.equal(p.pixelValues.length, ref.shape[0] * ref.shape[1]);
  let off = 0, worst = 0;
  ref.samples.forEach((v, k) => { const d = Math.abs(p.pixelValues[k * ref.every] - v); worst = Math.max(worst, d); if (d > 1e-6) off++; });
  let sum = 0, sumsq = 0;
  for (const x of p.pixelValues) { sum += x; sumsq += x * x; }
  console.log(`image ${width}×${height} → grid [${p.grid}], ${p.tokens} tokens: ${off} of ${ref.samples.length} samples off (max ${worst.toExponential(2)}); sum ${sum.toFixed(3)} vs ${ref.sum.toFixed(3)}`);
  assert.ok(worst <= 2 / 255 + 1e-6, `pixel off by ${worst}, more than one uint8 step`);
  assert.ok(off <= ref.samples.length * 0.002, `${off} samples off`);
  assert.ok(Math.abs(sum - ref.sum) <= 1e-3 * p.pixelValues.length * (2 / 255), `sum ${sum} vs ${ref.sum}`);
  passed++;
}
{ // placeholders and greedy sampling
  assert.deepEqual(expandPlaceholders([1, 9, 2, 9, 3], 9, [2, 1]), [1, 9, 9, 2, 9, 3]);
  assert.throws(() => expandPlaceholders([1, 9], 9, [1, 1]), /placeholders/);
  assert.equal(sampleToken(Float32Array.of(0, 5, 3), {}, [], [1]), 2, 'suppressed ids never win');
  assert.equal(sampleToken(Float32Array.of(0, 5, 4.9), { repetition_penalty: 1.1 }, [1]), 2, 'repetition penalty');
  passed++;
}
{ // the audio cap: 30 s per clip passes, a sample more is refused with the length in the message
  checkAudio(new Float32Array(MAX_AUDIO_SECONDS * 16000));
  assert.throws(() => checkAudio(new Float32Array(MAX_AUDIO_SECONDS * 16000 + 1), 2), /audio 2 is 30\.0 s long; the limit is 30 s/);
  assert.throws(() => checkAudio([0, 0]), /Float32Array/);
  passed++;
}
{ // fetchCached with an in-memory Cache Storage stand-in: a miss downloads once and stores the bytes; a hit reads
  // them back without a request; force downloads again; a stored copy cut short is replaced; delete is by URL
  const store = new Map();
  const cache = {
    match: async (u) => store.has(u) ? new Response(store.get(u).body, { headers: { 'content-length': String(store.get(u).len) } }) : undefined,
    put: async (u, r) => { store.set(u, { body: new Uint8Array(await r.arrayBuffer()), len: Number(r.headers.get('content-length')) }); },
    delete: async (u) => store.delete(u),
  };
  const data = Uint8Array.from({ length: 300000 }, (_, i) => (i * 7) & 255);
  let requests = 0;
  const fetchImpl = async () => { requests++; return new Response(new Blob([data]).stream(), { headers: { 'content-length': String(data.length) } }); };
  const events = [];
  const a = await fetchCached('u/x', { cache, fetchImpl, onProgress: (e) => events.push(e) });
  assert.deepEqual(a, data); assert.equal(requests, 1); assert.deepEqual(store.get('u/x').body, data);
  assert.equal(events.at(-1).status, 'download'); assert.equal(events.at(-1).loaded, data.length);
  events.length = 0;
  const b = await fetchCached('u/x', { cache, fetchImpl, onProgress: (e) => events.push(e) });
  assert.deepEqual(b, data); assert.equal(requests, 1, 'a hit makes no request'); assert.equal(events.at(-1).status, 'cache');
  await fetchCached('u/x', { cache, fetchImpl, force: true });
  assert.equal(requests, 2, 'force downloads again');
  store.set('u/x', { body: data.slice(0, 1000), len: data.length });
  assert.deepEqual(await fetchCached('u/x', { cache, fetchImpl }), data); assert.equal(requests, 3, 'a short copy is replaced');
  assert.deepEqual(store.get('u/x').body, data);
  assert.ok(await cache.delete('u/x')); assert.equal(store.size, 0);
  assert.deepEqual(await fetchCached('u/y', { fetchImpl }), data, 'no cache: a plain download');
  passed++;
}
const so = JSON.parse(readFileSync(new URL('speech-out.json', dir)));
{ // inverse STFT: the codec rows of the reference's first frames to numpy's wave, within 1e-4 of its peak; fed in
  // uneven pieces it gives the same samples
  const rows = new Float32Array(bin(so.istft.rows_file)), ref = new Float32Array(bin(so.istft.wave_file));
  const { n_fft, hop_length } = so.istft, width = n_fft + 2;
  const t0 = performance.now();
  const one = new Istft(n_fft, hop_length), whole = concatF([one.push(rows), one.end()]);
  const ms = performance.now() - t0;
  assert.equal(whole.length, ref.length, 'rows · hop samples');
  let worst = 0, peak = 0;
  for (let i = 0; i < ref.length; i++) { worst = Math.max(worst, Math.abs(whole[i] - ref[i])); peak = Math.max(peak, Math.abs(ref[i])); }
  console.log(`istft ${rows.length / width} rows → ${whole.length} samples in ${ms.toFixed(0)} ms: max |diff| ${worst.toExponential(2)} of peak ${peak.toFixed(4)}`);
  assert.ok(worst <= 1e-4 * peak, `istft max |diff| ${worst}`);
  const parts = new Istft(n_fft, hop_length), out = [];
  for (const [a, b] of [[0, 1], [1, 7], [7, 13], [13, rows.length / width]]) out.push(parts.push(rows.subarray(a * width, b * width)));
  out.push(parts.end());
  assert.deepEqual(concatF(out), whole, 'streamed iSTFT = offline');
  passed++;
}
{ // SpeechStream: with a fake causal codec (each row depends on its own frame only) streamed chunks equal one decode
  const K = 12, width = 1282, frames = so.frames.slice(0, 7);
  const codec = async (codes, F) => {
    const r = new Float32Array(6 * F * width);
    for (let f = 0; f < F; f++) for (let j = 0; j < 6; j++) for (let x = 0; x < width; x++) r[((f * 6) + j) * width + x] = Math.sin(Number(codes[(x % K) * F + f]) * 0.001 + x * 0.01 + j) * (x < 641 ? 0.5 : 3);
    return r;
  };
  const a = new SpeechStream(codec), b = new SpeechStream(codec), got = [];
  for (const [i, f] of frames.entries()) { a.push(f); b.push(f); if (i % 3 === 2) got.push(await a.decode(false)); }
  got.push(await a.decode(true));
  const once = await b.decode(true);
  assert.equal(once.length, frames.length * 1920);
  assert.deepEqual(concatF(got), once, 'chunked speech = one decode');
  passed++;
}
{ // MaskGIT: 3 passes in the order's schedule, each sees the codes revealed so far; an EOS code makes an EOS frame;
  // the repetition penalty reads each codebook's own history column
  const seen = [];
  const depth = async (h, codes, masked) => {
    seen.push([Array.from(codes, Number), Array.from(masked)]);
    const l = new Float32Array(12 * 2017);
    for (let k = 0; k < 12; k++) { l[k * 2017 + 100 + k] = 5; l[k * 2017 + 200 + k] = 4.9; }
    return l;
  };
  const order = [9, 2, 7, 4, 5, 11, 0, 3, 6, 10, 8, 1];
  const f = await sampleFrame(new Float32Array(4), depth, { order, sampling: audioSampling({ audio_temperature: 0, audio_repetition_penalty: 1.2 }), history: [Array.from({ length: 12 }, (_, k) => k === 3 ? 103 : 0)] });
  assert.equal(seen.length, 3);
  assert.deepEqual(seen[1][1], Array.from({ length: 12 }, (_, k) => [9, 2, 7, 4].includes(k) ? 0 : 1), 'pass 2 sees the first 4 revealed');
  assert.equal(f[9], 109); assert.equal(f[3], 203, 'codebook 3 penalised its repeat');
  const eos = await sampleFrame(new Float32Array(4), async () => { const l = new Float32Array(12 * 2017); l[5 * 2017 + 2016] = 9; return l; }, { order });
  assert.ok(eos.every((c) => c === 2016), 'any EOS code → an all-EOS frame');
  assert.deepEqual(audioSampling({ audio_temperature: 0.4 }), { do_sample: true, temperature: 0.4, top_k: 0, repetition_penalty: 1 });
  assert.equal(audioSampling({}).do_sample, false);
  assert.deepEqual([...randomOrder(12, () => 0.5)].sort((x, y) => x - y), Array.from({ length: 12 }, (_, i) => i));
  passed++;
}

// prompt ids: the engine's prepare() with transformers.js's tokenizer and the model's template
const model = process.env.LFM3_OMNI_DIR || '/Volumes/Models2/Models/lfm3/omni-2b-onnx';
let tf = null;
try {
  const spec = process.env.TRANSFORMERS_JS;
  tf = spec ? await import(pathToFileURL(existsSync(spec + '/package.json') ? `${spec}/dist/transformers.node.mjs` : spec).href) : await import('@huggingface/transformers');
} catch { tf = null; }
if (!tf || !existsSync(`${model}/tokenizer.json`)) {
  console.log(`prompt ids: SKIPPED (${!tf ? 'no transformers.js: set TRANSFORMERS_JS' : `no tokenizer files in ${model}: set LFM3_OMNI_DIR`})`);
} else {
  const read = (f) => readFileSync(`${model}/${f}`, 'utf8');
  const tc = JSON.parse(read('tokenizer_config.json'));
  const tokenizer = new tf.PreTrainedTokenizer(JSON.parse(read('tokenizer.json')), { ...tc, chat_template: read('chat_template.jinja') });
  const m = new Lfm3Omni({ ort: null, tokenizer, config: JSON.parse(read('config.json')), gen: {}, processor: JSON.parse(read('processor_config.json')), vision: null, audio: null, decoder: null });
  const rgb = new Uint8Array(bin(fx.image.file));
  for (const c of fx.cases) {
    const msgs = c.messages.map((msg) => ({ ...msg, content: msg.content.map((part) => part.type === 'image' ? { type: 'image', image: { data: rgb, width: fx.image.width, height: fx.image.height, channels: 3 } } : part.type === 'audio' ? { type: 'audio', audio: wave } : part) }));
    const { prompt, ids } = await m.prepare(msgs, { enable_thinking: c.enable_thinking });
    assert.equal(prompt, c.prompt, `${c.name}: template text`);
    assert.deepEqual(ids, c.prompt_ids, `${c.name}: prompt ids`);
    console.log(`prompt ids ${c.name}: ${ids.length} identical`);
  }
  const sp = await m.prepare([{ role: 'system', content: SPEECH_SYSTEM_PROMPT }, { role: 'user', content: [{ type: 'audio', audio: wave }] }]);
  assert.deepEqual([...sp.ids, m.tokens.mixed_start], so.prompt_ids, 's2s prompt ids (system turn, <|mixed_start|> appended)');
  console.log(`prompt ids s2s: ${so.prompt_ids.length} identical`);
  const think = await m.prepare([{ role: 'user', content: 'Hi' }], { enable_thinking: true });
  assert.ok(think.prompt.endsWith('<|im_start|>assistant\n<think>'), 'enable_thinking opens a <think> block');
  passed++;
}
console.log(`lfm3-omni: ${passed} checks passed`);

function concatF(parts) { const o = new Float32Array(parts.reduce((a, p) => a + p.length, 0)); let i = 0; for (const p of parts) { o.set(p, i); i += p.length; } return o; }
