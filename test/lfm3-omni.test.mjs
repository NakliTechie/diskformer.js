// engines/lfm3_omni.js's pure-JS parts against scripts/lfm3-omni-ref.py's fixture (test/refs/lfm3-omni): log-mel
// features, image preprocessing (resize, normalize, patches), and the prompt ids (chat template, placeholders).
//   node test/lfm3-omni.test.mjs
// The prompt-id check needs the model's tokenizer files (LFM3_OMNI_DIR, default the Models2 copy) and transformers.js
// 4.3.2 (TRANSFORMERS_JS: its package folder or entry file, else the `@huggingface/transformers` package); without
// them it says so and skips that part.
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { Lfm3Omni, logMel, melFrames, speechTokens, MEL, preprocessImage, imageConfig, expandPlaceholders, sampleToken, smartResize } from '../engines/lfm3_omni.js';

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
  const think = await m.prepare([{ role: 'user', content: 'Hi' }], { enable_thinking: true });
  assert.ok(think.prompt.endsWith('<|im_start|>assistant\n<think>'), 'enable_thinking opens a <think> block');
  passed++;
}
console.log(`lfm3-omni: ${passed} checks passed`);
