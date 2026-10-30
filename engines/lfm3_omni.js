/* lfm3_omni.js — LiquidAI LFM3-Omni-2B (text, image and speech in; text out) on onnxruntime-web's WebGPU EP.
 *
 * transformers.js 4.3.2 has no `lfm3_omni` model class, so this module runs Liquid's ONNX export by hand. It follows
 * Liquid's own Python runner (Liquid4All/onnx-export @ 5167e85: liquidonnx/lfm3_omni/session.py on top of
 * lfm3_audio/session.py and lfm3_vl/session.py), and is gated against scripts/lfm3-omni-ref.py (that runner on
 * onnxruntime's CPU EP): test/lfm3-omni.test.mjs checks the pure-JS parts, test/browser/lfm3-omni.html the tokens.
 *
 *   const m = await Lfm3Omni.load({ baseUrl: 'https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX/resolve/main/' });
 *   const out = await m.generate(
 *     [{ role: 'user', content: [{ type: 'image', image }, { type: 'audio', audio }, { type: 'text', text: 'Hi' }] }],
 *     { max_new_tokens: 512, do_sample: false, enable_thinking: false, tools, signal, onText: (delta) => … });
 *   // out = { text, stopReason: 'eos' | 'length' | 'aborted', tokens, reasoning, content }
 *   await m.dispose();
 *
 * Inputs: `image` an ImageData, { data: Uint8Array RGB or RGBA, width, height }, or anything drawImage takes; `audio`
 * a Float32Array of mono 16 kHz samples (toMono16k() resamples an AudioBuffer). Both may also come in order from
 * opts.images / opts.audios, as a transformers.js processor takes them beside the text.
 *
 * Graphs (the model card's WebGPU pairing): decoder_q4f16 (fp16 activations and caches, int4 weights),
 * vision_encoder_q8 and the float32 audio_encoder. Per prompt:
 *   chat template → ids; each <|image|> repeats once per image token (grid h·w / merge²), each <|audio|> once per
 *   speech token (ceil(mel frames / 8))
 *   image: antialiased bicubic resize to multiples of 32 (smart_resize) in torch's uint8 fixed point,
 *     (x/255 − 0.5)/0.5, 16×16 patches × 2 frames, block-major → vision_encoder → features [tokens, 2048]
 *   audio: log-mel (400-point periodic Hann, hop 160, 128 slaney bins, log10 ≥ −12, (x + 4)/4) → audio_encoder
 *   decoder prefill in chunks of ≤ 1,024 ids (each chunk takes the feature rows of its own placeholders), then one
 *     id per step; the decoder scatters the rows over the placeholders itself.
 *
 * Text slots follow lfm3_audio.session.generate_ids in TEXT mode: the placeholders (<|audio|>, <|audio_out|>,
 * <|image|>, <|video|>), <|mixed_start|> and <|audio_start|> are suppressed; <|im_end|> or <|text_end|> ends the turn.
 * Speech output (layer 2, not built): the step already returns the last hidden state, which the depth decoder
 * (MaskGIT, 12 codes per frame) reads, and takes `audio_out_codes`, the codes of the frame fed back as <|audio_out|>;
 * output_modalities: ['text', 'audio'] will switch generate() to lfm3_audio's TEXT / AUDIO / INTERLEAVED machine.
 *
 * Model files go to Cache Storage ('diskformer-models', keyed by URL) on the first load and come from there later;
 * load({ force: true }) downloads again, Lfm3Omni.deleteCache({ baseUrl }) removes this model's entries by name.
 * Audio parts are capped at MAX_AUDIO_SECONDS (30 s) per clip.
 *
 * The decoder sits behind Lfm3DecoderStep (step(ids, side) → { logits, hidden }, reset(), dispose()), so a WGSL
 * decoder can replace the ORT one while the encoders stay on ORT.
 */

export const ORT_WEB_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
export const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/+esm';
export const LFM3_OMNI_2B = {
  repo: 'LiquidAI/LFM3-Omni-2B-ONNX',
  baseUrl: 'https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX/resolve/main/',
  graphs: { decoder: 'onnx/decoder_q4f16.onnx', vision: 'onnx/vision_encoder_q8.onnx', audio: 'onnx/audio_encoder.onnx' },
  json: ['config.json', 'generation_config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'],
};
// The model card: prefill at most this many ids per decoder run
export const PREFILL_CHUNK = 1024;

// ── Speech in: log-mel features (lfm3_audio.session.log_mel, Lfm3AudioFeatureExtractor) ───────────────────────────
export const MEL = { sampling_rate: 16000, num_mel_bins: 128, n_fft: 400, win_length: 400, hop_length: 160,
  log_floor: -12, normalization_offset: 4, normalization_divisor: 4, encoder_stride: 8 };

export function melConfig(fe = {}) {
  const m = { ...MEL };
  for (const k of Object.keys(MEL)) if (fe[k] !== undefined) m[k] = fe[k];
  if (fe.feature_size !== undefined) m.num_mel_bins = fe.feature_size;
  return m;
}
const mod = (a, n) => ((a % n) + n) % n;
export function melFrames(samples, mel = MEL) {
  const padded = Math.max(mel.n_fft, samples + mod(-(samples - mel.n_fft), mel.hop_length));
  return 1 + Math.floor((padded - mel.n_fft) / mel.hop_length);
}
export const speechTokens = (frames, mel = MEL) => Math.ceil(frames / mel.encoder_stride);
// Longest speech clip per audio part, in seconds (the NakliOS assistant's turn limit)
export const MAX_AUDIO_SECONDS = 30;
export function checkAudio(wave, index = 0, maxSeconds = MAX_AUDIO_SECONDS, rate = MEL.sampling_rate) {
  if (!(wave instanceof Float32Array)) throw new TypeError(`audio ${index}: expected a Float32Array of 16 kHz mono samples`);
  if (wave.length > maxSeconds * rate) throw new RangeError(`audio ${index} is ${(wave.length / rate).toFixed(1)} s long; the limit is ${maxSeconds} s per clip`);
}

// [n_fft/2 + 1][num_mel_bins] as a flat Float32Array (bin-major): transformers.audio_utils.mel_filter_bank with the
// slaney scale and norm from 0 to sampling_rate / 2, rounded to float32 as the Python side does.
export function melFilters(mel = MEL) {
  const bins = mel.n_fft / 2 + 1, n = mel.num_mel_bins;
  const toMel = (f) => (f >= 1000 ? 15 + Math.log(f / 1000) * (27 / Math.log(6.4)) : (3 * f) / 200);
  const toHz = (m) => (m >= 15 ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)) : (200 * m) / 3);
  const lo = toMel(0), hi = toMel(mel.sampling_rate / 2);
  const f = Array.from({ length: n + 2 }, (_, i) => toHz(lo + ((hi - lo) * i) / (n + 1)));
  const out = new Float32Array(bins * n);
  for (let b = 0; b < bins; b++) {
    const hz = ((mel.sampling_rate / 2) * b) / (bins - 1);
    for (let j = 0; j < n; j++) {
      const down = -(f[j] - hz) / (f[j + 1] - f[j]), up = (f[j + 2] - hz) / (f[j + 2] - f[j + 1]);
      out[b * n + j] = Math.max(0, Math.min(down, up)) * (2 / (f[j + 2] - f[j]));
    }
  }
  return out;
}

// input_features [num_mel_bins, T] (row-major Float32Array) of one mono 16 kHz clip: right-padded, |DFT|² of
// center=False frames under a periodic Hann (float32 window, float64 math), mel filters, log10, clamp, normalize.
export function logMel(wave, mel = MEL, filters = melFilters(mel)) {
  const N = mel.n_fft, hop = mel.hop_length, bins = N / 2 + 1, nm = mel.num_mel_bins;
  const T = melFrames(wave.length, mel);
  const padded = new Float64Array((T - 1) * hop + N);
  for (let i = 0; i < wave.length; i++) padded[i] = Math.fround(wave[i]);
  const win = new Float64Array(N), off = (N - mel.win_length) >> 1;
  for (let i = 0; i < mel.win_length; i++) win[off + i] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / mel.win_length));
  const cos = new Float64Array(N), sin = new Float64Array(N);
  for (let i = 0; i < N; i++) { cos[i] = Math.cos((2 * Math.PI * i) / N); sin[i] = Math.sin((2 * Math.PI * i) / N); }
  const out = new Float32Array(nm * T), seg = new Float64Array(N), power = new Float64Array(bins), acc = new Float64Array(nm);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < N; i++) seg[i] = padded[t * hop + i] * win[i];
    for (let k = 0; k < bins; k++) {
      let re = 0, im = 0;
      for (let i = 0, a = 0; i < N; i++, a = (a + k) % N) { re += seg[i] * cos[a]; im -= seg[i] * sin[a]; }
      power[k] = re * re + im * im;
    }
    acc.fill(0);
    for (let k = 0; k < bins; k++) { const p = power[k]; if (p) for (let j = 0; j < nm; j++) acc[j] += filters[k * nm + j] * p; }
    for (let j = 0; j < nm; j++) {
      const v = Math.max(acc[j] > 0 ? Math.log10(acc[j]) : -Infinity, mel.log_floor);
      out[j * T + t] = (v + mel.normalization_offset) / mel.normalization_divisor;
    }
  }
  return { data: out, dims: [nm, T] };
}

// Mono 16 kHz Float32Array of a decoded AudioBuffer (browser; OfflineAudioContext's resampler, not soxr).
export async function toMono16k(buffer, rate = MEL.sampling_rate) {
  const ctx = new OfflineAudioContext(1, Math.ceil(buffer.duration * rate), rate);
  const src = ctx.createBufferSource(); src.buffer = buffer; src.connect(ctx.destination); src.start();
  return (await ctx.startRendering()).getChannelData(0).slice();
}

// ── Images: Qwen2VLImageProcessor (transformers 5.19, its torchvision backend) as preprocessor_config.json sets it ──
export function imageConfig(p = {}) {
  return { patch_size: p.patch_size ?? 16, merge_size: p.merge_size ?? 2, temporal_patch_size: p.temporal_patch_size ?? 2,
    min_pixels: p.size?.shortest_edge ?? 65536, max_pixels: p.size?.longest_edge ?? 2097152,
    mean: p.image_mean ?? [0.5, 0.5, 0.5], std: p.image_std ?? [0.5, 0.5, 0.5], rescale: p.rescale_factor ?? 1 / 255 };
}

// Python's round(): halves to even
const roundEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };
export function smartResize(height, width, factor, minPixels, maxPixels) {
  if (Math.max(height, width) / Math.min(height, width) > 200) throw new Error(`aspect ratio over 200: ${width}×${height}`);
  let h = roundEven(height / factor) * factor, w = roundEven(width / factor) * factor;
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.max(factor, Math.floor(height / beta / factor) * factor); w = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil((height * beta) / factor) * factor; w = Math.ceil((width * beta) / factor) * factor;
  }
  return [h, w];
}

// torch's antialiased bicubic weights (a = −0.5; support 2 × scale when shrinking), align_corners=False
function aaWeights(inSize, outSize) {
  const scale = inSize / outSize, support = scale >= 1 ? 2 * scale : 2, inv = scale >= 1 ? 1 / scale : 1;
  const cubic = (x) => { x = Math.abs(x); return x < 1 ? (1.5 * x - 2.5) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * -0.5 : 0; };
  const rows = [];
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(Math.trunc(center - support + 0.5), 0), hi = Math.min(Math.trunc(center + support + 0.5), inSize);
    const w = []; let total = 0;
    for (let j = lo; j < hi; j++) { const v = cubic((j - center + 0.5) * inv); w.push(v); total += v; }
    rows.push({ lo, w: w.map((v) => (total ? v / total : 0)) });
  }
  return rows;
}

// The weights of aaWeights as int16 fixed point: torch's uint8 antialias kernel (UpSampleKernel's
// _compute_index_ranges_int16_weights, Pillow's scheme) takes the most fraction bits that keep the largest weight
// under 2^15, and rounds each weight away from zero.
function int16Weights(rows) {
  let wmax = 0;
  for (const r of rows) for (const v of r.w) wmax = Math.max(wmax, Math.abs(v));
  let bits = 0;
  for (; bits < 22; bits++) if (Math.trunc(0.5 + wmax * 2 ** (bits + 1)) >= 2 ** 15) break;
  return { bits, rows: rows.map(({ lo, w }) => ({ lo, w: w.map((v) => Math.trunc(v < 0 ? -0.5 + v * 2 ** bits : 0.5 + v * 2 ** bits)) })) };
}
// One pass along an axis in integers: Σ w·x + half, shifted down by the fraction bits, clamped to 0..255.
function pass(src, n, rows, bits, read, write) {
  const half = 2 ** (bits - 1), div = 2 ** bits;
  for (let o = 0; o < rows.length; o++) {
    const { lo, w } = rows[o];
    for (let line = 0; line < n; line++) {
      let s = half;
      for (let k = 0; k < w.length; k++) s += w[k] * src[read(line, lo + k)];
      write(line, o, Math.min(255, Math.max(0, Math.floor(s / div))));
    }
  }
}

// Planar uint8-valued Float32Array [3][nh][nw] of interleaved uint8 pixels (channels 3 or 4), as torchvision resizes
// a uint8 tensor with bicubic antialias (torch's uint8 path): width pass to uint8, then height pass to uint8.
export function resizeBicubic(data, width, height, channels, nw, nh) {
  const out = new Float32Array(3 * nh * nw);
  let tmp = new Uint8Array(3 * height * nw);
  if (nw === width) { for (let c = 0; c < 3; c++) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) tmp[(c * height + y) * nw + x] = data[(y * width + x) * channels + c]; }
  else {
    const { bits, rows } = int16Weights(aaWeights(width, nw));
    for (let c = 0; c < 3; c++) pass(data, height, rows, bits, (y, x) => (y * width + x) * channels + c, (y, x, v) => { tmp[(c * height + y) * nw + x] = v; });
  }
  if (nh === height) { for (let i = 0; i < out.length; i++) out[i] = tmp[i]; return out; }
  const { bits, rows } = int16Weights(aaWeights(height, nh));
  for (let c = 0; c < 3; c++) pass(tmp, nw, rows, bits, (x, y) => (c * height + y) * nw + x, (x, y, v) => { out[(c * nh + y) * nw + x] = v; });
  return out;
}

// { pixelValues: Float32Array [gh·gw, 3·T·p·p] (block-major rows, the vision encoder's [gh/m, gw/m, m, m, dim]),
//   grid: [1, gh, gw], tokens: gh·gw / m² } of one RGB(A) image
export function preprocessImage({ data, width, height, channels }, cfg = imageConfig()) {
  channels = channels ?? (data.length === width * height * 4 ? 4 : 3);
  const p = cfg.patch_size, m = cfg.merge_size, T = cfg.temporal_patch_size;
  const [nh, nw] = smartResize(height, width, p * m, cfg.min_pixels, cfg.max_pixels);
  const px = resizeBicubic(data, width, height, channels, nw, nh);
  for (let c = 0; c < 3; c++) for (let i = c * nh * nw; i < (c + 1) * nh * nw; i++) px[i] = (px[i] * cfg.rescale - cfg.mean[c]) / cfg.std[c];
  const gh = nh / p, gw = nw / p, dim = 3 * T * p * p, pv = new Float32Array(gh * gw * dim);
  let row = 0;
  for (let bh = 0; bh < gh / m; bh++) for (let bw = 0; bw < gw / m; bw++) for (let mh = 0; mh < m; mh++) for (let mw = 0; mw < m; mw++, row++) {
    const y0 = (bh * m + mh) * p, x0 = (bw * m + mw) * p;
    let o = row * dim;
    for (let c = 0; c < 3; c++) for (let t = 0; t < T; t++) for (let y = 0; y < p; y++) {
      const base = (c * nh + y0 + y) * nw + x0;
      for (let x = 0; x < p; x++) pv[o++] = px[base + x];
    }
  }
  return { pixelValues: pv, grid: [1, gh, gw], tokens: (gh * gw) / (m * m), dims: [gh / m, gw / m, m, m, dim] };
}

// RGB(A) pixels of an image input: raw { data, width, height } as is, else drawn through an OffscreenCanvas (RGBA)
async function pixelsOf(image) {
  if (image && image.data && image.width && image.height) return image;
  const bmp = typeof createImageBitmap === 'function' && !(image instanceof ImageBitmap) ? await createImageBitmap(image) : image;
  const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext('2d');
  g.drawImage(bmp, 0, 0);
  const d = g.getImageData(0, 0, bmp.width, bmp.height);
  return { data: d.data, width: d.width, height: d.height, channels: 4 };
}

// ── Prompt ids ────────────────────────────────────────────────────────────────────────────────────────────────────
// ids with the n-th occurrence of `token` repeated counts[n] times (Lfm3AudioProcessor, expand_image_tokens)
export function expandPlaceholders(ids, token, counts) {
  const found = ids.filter((t) => t === token).length;
  if (found !== counts.length) throw new Error(`prompt has ${found} placeholders ${token}, got ${counts.length} inputs`);
  const out = []; let n = 0;
  for (const t of ids) if (t === token) for (let k = counts[n++]; k > 0; k--) out.push(t); else out.push(t);
  return out;
}

// Messages for the chat template: media parts keep only their type (the template writes the placeholder);
// the media themselves, in prompt order, from the parts or else from images / audios.
export function splitMedia(messages, images = [], audios = []) {
  const media = { images: [], audios: [] }; let ni = 0, na = 0;
  const strip = messages.map((msg) => {
    if (!Array.isArray(msg.content)) return msg;
    return { ...msg, content: msg.content.map((part) => {
      if (part?.type === 'image') { media.images.push(part.image ?? images[ni]); ni++; return { type: 'image' }; }
      if (part?.type === 'audio') { media.audios.push(part.audio ?? audios[na]); na++; return { type: 'audio' }; }
      return part;
    }) };
  });
  for (const [k, list] of Object.entries(media)) list.forEach((x, i) => { if (!x) throw new Error(`no data for ${k.slice(0, -1)} ${i}`); });
  return { messages: strip, ...media };
}

// ── Sampling (lfm3_audio.session.sample_token) ──────────────────────────────────────────────────────────────────────
export function mulberry32(seed) {
  return () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// One id from logits: repetition penalty over history, suppressed ids to −inf, argmax unless sampling, else
// temperature, top-k, top-p, min-p and a draw from rand.
export function sampleToken(logits, { do_sample = false, temperature = 1, top_k = 0, top_p = 1, min_p = 0, repetition_penalty = 1 } = {}, history = [], suppress = [], rand = Math.random) {
  const s = Float32Array.from(logits);
  if (repetition_penalty !== 1) for (const t of new Set(history)) s[t] = s[t] < 0 ? s[t] * repetition_penalty : s[t] / repetition_penalty;
  for (const t of suppress) s[t] = -Infinity;
  if (!do_sample || temperature === 0) { let b = 0; for (let i = 1; i < s.length; i++) if (s[i] > s[b]) b = i; return b; }
  let ids = Array.from(s.keys()).filter((i) => s[i] > -Infinity).sort((a, b) => s[b] - s[a]);
  if (top_k > 0) ids = ids.slice(0, top_k);
  const mx = s[ids[0]] / temperature;
  let p = ids.map((i) => Math.exp(s[i] / temperature - mx)), z = p.reduce((a, b) => a + b, 0);
  p = p.map((x) => x / z);
  let keep = ids.length;
  if (top_p < 1) { let c = 0; keep = 0; while (keep < ids.length && c < top_p) c += p[keep++]; }
  if (min_p > 0) keep = Math.min(keep, p.filter((x) => x >= min_p * p[0]).length);
  ids = ids.slice(0, Math.max(1, keep)); p = p.slice(0, ids.length); z = p.reduce((a, b) => a + b, 0);
  let r = rand() * z;
  for (let k = 0; k < ids.length; k++) { r -= p[k]; if (r <= 0) return ids[k]; }
  return ids[ids.length - 1];
}

// ── The decoder step on ORT ──────────────────────────────────────────────────────────────────────────────────────
// Lfm3DecoderStep: what generate() needs from a decoder, whatever runs it.
//   step(ids, side) → { logits: Float32Array [vocab], hidden: Float32Array [hidden] } at the last id, after the ids so
//     far; side = { image_features, video_features, audio_features: Float32Array rows × hidden, one row per
//     placeholder among `ids`, in order; audio_out_codes: BigInt64Array rows × 12, one per <|audio_out|> }
//   past (ids so far), reset() (drops the caches), dispose()
export class OrtLfm3Decoder {
  constructor(ort, session, text) {
    this.ort = ort; this.session = session; this.hidden = text.hidden_size;
    this.inputs = new Set(session.inputNames);
    this.layers = text.layer_types.map((type, i) => ({ i, conv: type === 'conv' }));
    this.convShape = [1, text.hidden_size, text.conv_L_cache];
    this.kvShape = [1, text.num_key_value_heads, 0, text.head_dim];
    this.cache = null; this.past = 0;
  }
  static async create(ort, source, opts, text) {
    const gpu = {};
    text.layer_types.forEach((type, i) => { if (type === 'conv') gpu[`present_conv.${i}`] = 'gpu-buffer'; else { gpu[`present.${i}.key`] = 'gpu-buffer'; gpu[`present.${i}.value`] = 'gpu-buffer'; } });
    const session = await createSession(ort, source, { ...opts, preferredOutputLocation: opts.device === 'webgpu' ? gpu : undefined });
    return new OrtLfm3Decoder(ort, session, text);
  }
  #empty() {
    const T = this.ort.Tensor, cache = {};
    for (const { i, conv } of this.layers) {
      if (conv) cache[`past_conv.${i}`] = new T('float16', new Uint16Array(this.convShape.reduce((a, b) => a * b)), this.convShape);
      else for (const kv of ['key', 'value']) cache[`past_key_values.${i}.${kv}`] = new T('float16', new Uint16Array(0), this.kvShape);
    }
    return cache;
  }
  async step(ids, side = {}) {
    const T = this.ort.Tensor, n = ids.length, H = this.hidden;
    if (!this.cache) this.cache = this.#empty();
    const i64 = (arr) => BigInt64Array.from(arr, (x) => BigInt(x));
    const rows = (x) => (x && x.length ? x : new Float32Array(0));
    const feeds = {
      input_ids: new T('int64', i64(ids), [1, n]),
      attention_mask: new T('int64', new BigInt64Array(this.past + n).fill(1n), [1, this.past + n]),
      position_ids: new T('int64', i64(Array.from({ length: n }, (_, k) => this.past + k)), [1, n]),
      logits_to_keep: new T('int64', BigInt64Array.of(1n), [1]),
      ...this.cache,
    };
    for (const name of ['image_features', 'video_features', 'audio_features']) {
      const x = rows(side[name]);
      if (this.inputs.has(name)) feeds[name] = new T('float32', x, [x.length / H, H]);
    }
    const codes = side.audio_out_codes && side.audio_out_codes.length ? side.audio_out_codes : new BigInt64Array(0);
    if (this.inputs.has('audio_out_codes')) feeds.audio_out_codes = new T('int64', codes, [codes.length / 12, 12]);
    const out = await this.session.run(feeds);
    const next = {};
    for (const [name, t] of Object.entries(out)) {
      if (name.startsWith('present_conv.')) next[name.replace('present_conv.', 'past_conv.')] = t;
      else if (name.startsWith('present.')) next[name.replace('present.', 'past_key_values.')] = t;
    }
    for (const t of Object.values(this.cache)) t.dispose?.();
    this.cache = next; this.past += n;
    const logits = out.logits.data, hidden = out.hidden_states.data, V = logits.length / out.logits.dims[1];
    const res = { logits: logits.slice(logits.length - V), hidden: Float32Array.from(hidden.subarray(hidden.length - H)) };
    out.logits.dispose?.(); out.hidden_states.dispose?.();
    return res;
  }
  reset() { if (this.cache) for (const t of Object.values(this.cache)) t.dispose?.(); this.cache = null; this.past = 0; }
  async dispose() { this.reset(); await this.session.release(); }
}

async function createSession(ort, { model, data, dataPath }, { device = 'webgpu', preferredOutputLocation } = {}) {
  const ep = device === 'webgpu' ? { name: 'webgpu', enableMatmulFp32Accumulation: true } : device;
  return ort.InferenceSession.create(model, {
    executionProviders: [ep], graphOptimizationLevel: 'all',
    externalData: data ? [{ path: dataPath, data }] : undefined,
    ...(preferredOutputLocation ? { preferredOutputLocation } : {}),
  });
}

// ── Model files: Cache Storage (as kohra's fetchCached) ─────────────────────────────────────────────────────────────
// One Cache Storage cache for diskformer's ONNX engines; each entry is keyed by its file URL, so deleteCache() can
// remove one model's entries by name. (GGUF engines keep their weights in OPFS stores instead: src/ingest.js.)
export const CACHE_NAME = 'diskformer-models';

// Bytes of url: a Cache Storage hit is read from disk; a miss is downloaded once, its stream split between the cache
// (written to disk as it arrives) and one buffer here, so a 1.4 GB file is never held twice by this module. A
// failed put (quota) still returns the bytes; the next load downloads again. force: skip the stored copy and replace
// it. onProgress({ status: 'download' | 'cache', loaded, total }); fetchImpl for tests.
export async function fetchCached(url, { cache = null, force = false, onProgress = () => {}, fetchImpl = fetch } = {}) {
  let res = cache && !force ? await cache.match(url) : null;
  const hit = !!res;
  if (!res) {
    res = await fetchImpl(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  }
  const status = hit ? 'cache' : 'download';
  const total = Number(res.headers.get('content-length')) || 0;
  let body = res.body, stored = null;
  if (cache && !hit && body) {
    const [mine, theirs] = body.tee();
    body = mine;
    stored = cache.put(url, new Response(theirs, total ? { headers: { 'content-length': String(total) } } : {})).catch(() => {});
  }
  if (!body) { const b = new Uint8Array(await res.arrayBuffer()); onProgress({ status, loaded: b.length, total: b.length }); return b; }
  // straight into one buffer of the announced size; chunks only when the size is unknown or the body runs past it
  // (a compressed response announces the compressed length)
  const bytes = new Uint8Array(total), chunks = [], reader = body.getReader();
  let loaded = 0, filled = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!chunks.length && filled + value.length <= total) { bytes.set(value, filled); filled += value.length; } else chunks.push(value);
    loaded += value.length;
    onProgress({ status, loaded, total: Math.max(total, loaded) });
  }
  await stored;
  if (hit && total && loaded !== total) {        // a stored copy cut short: drop it and download again
    await cache.delete(url);
    return fetchCached(url, { cache, force: true, onProgress, fetchImpl });
  }
  if (!chunks.length && filled === total) return bytes;
  const out = new Uint8Array(loaded);
  out.set(bytes.subarray(0, filled)); let o = filled;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// ── The engine ────────────────────────────────────────────────────────────────────────────────────────────────────
export class Lfm3Omni {
  // opts: baseUrl (the repo layout) or files { 'onnx/decoder_q4f16.onnx': URL | Blob | ArrayBuffer | Uint8Array, … };
  // ort (onnxruntime-web/webgpu module), transformers (for PreTrainedTokenizer) or tokenizer; device 'webgpu';
  // onProgress({ status: 'download' | 'cache', file, loaded, total }) while a file arrives, then 'session' per graph;
  // cacheName (CACHE_NAME; null: no Cache Storage) and force (download again, replacing the stored copies)
  static async load(opts = {}) {
    const { baseUrl = LFM3_OMNI_2B.baseUrl, files = {}, device = 'webgpu', onProgress = () => {}, cacheName = CACHE_NAME, force = false } = opts;
    const src = (name) => files[name] ?? new URL(name, baseUrl.endsWith('/') ? baseUrl : baseUrl + '/').href;
    const cache = cacheName && globalThis.caches ? await caches.open(cacheName) : null;
    const bytes = async (name) => {
      const s = src(name);
      if (s instanceof Uint8Array) return s;
      if (s instanceof ArrayBuffer) return new Uint8Array(s);
      if (s instanceof Blob) return new Uint8Array(await s.arrayBuffer());
      return fetchCached(s, { cache, force, onProgress: (e) => onProgress({ ...e, file: name }) });
    };
    const json = async (name) => JSON.parse(new TextDecoder().decode(await bytes(name)));
    const text = async (name) => new TextDecoder().decode(await bytes(name));

    onProgress({ status: 'config' });
    const [config, gen, processor] = await Promise.all([json('config.json'), json('generation_config.json').catch(() => ({})), json('processor_config.json')]);
    if (config.model_type !== 'lfm3_omni') throw new Error(`not an LFM3-Omni config: ${config.model_type}`);
    let tokenizer = opts.tokenizer;
    if (!tokenizer) {
      onProgress({ status: 'tokenizer' });
      const tf = opts.transformers ?? (await import(TRANSFORMERS_URL));
      const [tj, tc, tpl] = await Promise.all([json('tokenizer.json'), json('tokenizer_config.json'), text('chat_template.jinja')]);
      tokenizer = new tf.PreTrainedTokenizer(tj, { ...tc, chat_template: tc.chat_template ?? tpl });
    }
    const ort = opts.ort ?? (await import(ORT_WEB_URL));
    // the graph and its external data, both read here (byte progress); the session is built from the bytes, which
    // are dropped once onnxruntime holds the weights
    const graph = async (role) => {
      const file = LFM3_OMNI_2B.graphs[role];
      const model = await bytes(file), data = await bytes(file + '_data');
      onProgress({ status: 'session', file });
      return { model, data, dataPath: file.split('/').pop() + '_data' };
    };
    const sessionOpts = { device };
    const vision = await createSession(ort, await graph('vision'), sessionOpts);
    const audio = await createSession(ort, await graph('audio'), sessionOpts);
    const decoder = await OrtLfm3Decoder.create(ort, await graph('decoder'), sessionOpts, config.text_config);
    onProgress({ status: 'ready' });
    return new Lfm3Omni({ ort, tokenizer, config, gen, processor, vision, audio, decoder });
  }

  // Every file load() reads, relative to baseUrl
  static files() {
    return [...LFM3_OMNI_2B.json, 'chat_template.jinja', ...Object.values(LFM3_OMNI_2B.graphs).flatMap((g) => [g, g + '_data'])];
  }
  // Stored copies of this model's files under baseUrl: [{ file, url, bytes }] (bytes from the stored length)
  static async cached({ baseUrl = LFM3_OMNI_2B.baseUrl, cacheName = CACHE_NAME } = {}) {
    if (!globalThis.caches) return [];
    const cache = await caches.open(cacheName), out = [];
    for (const file of Lfm3Omni.files()) {
      const url = new URL(file, baseUrl.endsWith('/') ? baseUrl : baseUrl + '/').href, r = await cache.match(url);
      if (r) out.push({ file, url, bytes: Number(r.headers.get('content-length')) || null });
    }
    return out;
  }
  // Deletes this model's stored files by name (other entries in the cache stay); resolves to the files deleted
  static async deleteCache({ baseUrl = LFM3_OMNI_2B.baseUrl, cacheName = CACHE_NAME } = {}) {
    if (!globalThis.caches) return [];
    const cache = await caches.open(cacheName), gone = [];
    for (const { file, url } of await Lfm3Omni.cached({ baseUrl, cacheName })) if (await cache.delete(url)) gone.push(file);
    return gone;
  }

  constructor({ ort, tokenizer, config, gen, processor, vision, audio, decoder }) {
    Object.assign(this, { ort, tokenizer, config, generationConfig: gen, vision, audio, decoder });
    // true from load until dispose
    this.ready = !!decoder;
    this.hidden = config.text_config.hidden_size;
    // ids per prefill run (the model card's cap; a test lowers it to cut placeholders across runs)
    this.prefillChunk = PREFILL_CHUNK;
    this.mel = melConfig(processor.feature_extractor);
    this.melFilters = melFilters(this.mel);
    this.image = imageConfig(processor.image_processor);
    const t = (k, d) => config[k] ?? d;
    this.tokens = {
      audio: t('audio_token_id', 125031), audio_out: t('audio_out_token_id', 125032), image: t('image_token_id', 125030),
      video: t('video_token_id', 125029), audio_start: t('audio_start_token_id', 125012), text_end: t('text_end_token_id', 125014),
      mixed_start: t('mixed_start_token_id', 125015), eos: config.text_config.eos_token_id,
      think_end: tokenizer.convert_tokens_to_ids?.('</think>') ?? null,
    };
    // the speech-output layer reads these (lfm3_audio.config): not used in text mode
    this.speech = { num_codebooks: t('num_codebooks', 12), codebook_size: t('codebook_size', 2017), audio_eos_code: t('audio_eos_code', 2016),
      maskgit_schedule: t('maskgit_schedule', [4, 4, 4]), interleaved_n_text: t('interleaved_n_text', 6), interleaved_n_audio: t('interleaved_n_audio', 12) };
  }

  // The chat template's text and ids (placeholders expanded) of messages; media come back for the encoders.
  async prepare(messages, { images = [], audios = [], enable_thinking = false, tools, add_generation_prompt = true, max_audio_seconds = MAX_AUDIO_SECONDS } = {}) {
    const m = splitMedia(messages, images, audios);
    m.audios.forEach((wave, i) => checkAudio(wave, i, max_audio_seconds, this.mel.sampling_rate));
    const prompt = this.tokenizer.apply_chat_template(m.messages, { tokenize: false, add_generation_prompt, enable_thinking, ...(tools ? { tools } : {}) });
    let ids = Array.from(this.tokenizer.encode(prompt, { add_special_tokens: false }), Number);
    const pics = [];
    for (const img of m.images) pics.push(preprocessImage(await pixelsOf(img), this.image));
    const mels = m.audios.map((wave) => logMel(wave, this.mel, this.melFilters));
    ids = expandPlaceholders(ids, this.tokens.audio, mels.map((x) => speechTokens(x.dims[1], this.mel)));
    ids = expandPlaceholders(ids, this.tokens.image, pics.map((x) => x.tokens));
    return { prompt, ids, pics, mels };
  }

  // image_features [Σ tokens × hidden] of preprocessed images: one vision run per image, in prompt order
  async encodeImages(pics) {
    const parts = [];
    for (const p of pics) {
      const out = await this.vision.run({ pixel_values: new this.ort.Tensor('float32', p.pixelValues, p.dims) });
      const f = out.image_features;
      if (f.dims[0] !== p.tokens) throw new Error(`grid ${p.grid} gave ${f.dims[0]} image features, expected ${p.tokens}`);
      parts.push(Float32Array.from(f.data)); f.dispose?.();
    }
    return concat(parts);
  }
  // audio_features [Σ tokens × hidden] of log-mel clips: one audio_encoder run per clip
  async encodeAudio(mels) {
    const parts = [];
    for (const m of mels) {
      const out = await this.audio.run({ input_features: new this.ort.Tensor('float32', m.data, [1, ...m.dims]) });
      const f = out.audio_features;
      if (f.dims[0] !== speechTokens(m.dims[1], this.mel)) throw new Error(`${m.dims[1]} mel frames gave ${f.dims[0]} audio features`);
      parts.push(Float32Array.from(f.data)); f.dispose?.();
    }
    return concat(parts);
  }

  // Prefill ids in chunks of ≤ prefillChunk; each chunk takes the feature rows of its own placeholders.
  async prefill(ids, side) {
    const H = this.hidden, at = { image_features: 0, audio_features: 0 }, token = { image_features: this.tokens.image, audio_features: this.tokens.audio };
    let last;
    for (let s = 0; s < ids.length; s += this.prefillChunk) {
      const chunk = ids.slice(s, s + this.prefillChunk), feed = {};
      for (const k of Object.keys(at)) {
        const n = chunk.filter((t) => t === token[k]).length;
        feed[k] = side[k] ? side[k].subarray(at[k] * H, (at[k] + n) * H) : undefined;
        at[k] += n;
      }
      last = await this.decoder.step(chunk, feed);
    }
    for (const k of Object.keys(at)) if ((side[k]?.length ?? 0) !== at[k] * H) throw new Error(`${at[k]} placeholders for ${(side[k]?.length ?? 0) / H} ${k} rows`);
    return last;
  }

  // Answer to messages, with transformers.js-style options (max_new_tokens, do_sample, temperature, top_k, top_p,
  // min_p, repetition_penalty, enable_thinking, tools, seed). Resolves { text, stopReason, tokens, reasoning, content }:
  // text is the raw answer (special tokens skipped; <think> and <tool_call> markup kept, the caller parses it),
  // stopReason 'eos' | 'length' | 'aborted'. Streaming: onText(delta) with each new piece of text (held back while
  // a token ends inside a UTF-8 character), onToken(id), and a transformers.js streamer (put([[id]]), end()). signal
  // (AbortSignal) is read before every decode step; an abort resolves, it does not throw.
  async generate(messages, opts = {}) {
    const { max_new_tokens = 1024, signal, onText, onToken, streamer, seed, output_modalities = ['text'] } = opts;
    if (output_modalities.includes('audio')) throw new Error('LFM3-Omni speech output is not built yet (text only)');
    const sampling = { do_sample: opts.do_sample ?? false, temperature: opts.temperature ?? 1, top_k: opts.top_k ?? 0,
      top_p: opts.top_p ?? 1, min_p: opts.min_p ?? 0, repetition_penalty: opts.repetition_penalty ?? 1 };
    const rand = seed === undefined ? Math.random : mulberry32(seed);
    const T = this.tokens, tokens = [], history = [];
    const suppress = [T.audio, T.audio_out, T.image, T.video, T.mixed_start, T.audio_start];
    let stopReason = 'length', sent = 0;
    const emit = () => {
      const full = this.tokenizer.decode(tokens, { skip_special_tokens: true });
      if (full.length > sent && !full.endsWith('\uFFFD')) { onText(full.slice(sent)); sent = full.length; }
    };
    if (signal?.aborted) return { text: '', stopReason: 'aborted', tokens, reasoning: '', content: '' };
    const { ids, pics, mels } = await this.prepare(messages, opts);
    const side = { image_features: pics.length ? await this.encodeImages(pics) : undefined, audio_features: mels.length ? await this.encodeAudio(mels) : undefined };
    try {
      let { logits } = await this.prefill(ids, side);
      // TEXT mode of lfm3_audio's slot machine; the speech layer adds AUDIO / INTERLEAVED slots here, read the step's
      // hidden state, and feeds frames back as <|audio_out|> with audio_out_codes
      for (let step = 0; step < max_new_tokens; step++) {
        if (signal?.aborted) { stopReason = 'aborted'; break; }
        const tok = sampleToken(logits, sampling, history, suppress, rand);
        if (tok === T.eos || tok === T.text_end) { stopReason = 'eos'; break; }
        tokens.push(tok); history.push(tok); if (history.length > 32) history.shift();
        streamer?.put?.([[BigInt(tok)]]); onToken?.(tok);
        if (onText) emit();
        if (step === max_new_tokens - 1) break;
        ({ logits } = await this.decoder.step([tok]));
      }
    } finally { this.decoder.reset(); streamer?.end?.(); }
    const text = this.tokenizer.decode(tokens, { skip_special_tokens: true });
    if (onText && text.length > sent) onText(text.slice(sent));
    return { text, stopReason, tokens, ...this.decodeText(tokens) };
  }

  // Greedy ids after prompt ids (the gate's comparison): no template, no stop on EOS unless asked.
  async greedyIds(ids, side = {}, n = 32, { stopAtEos = true } = {}) {
    const T = this.tokens, suppress = [T.audio, T.audio_out, T.image, T.video, T.mixed_start, T.audio_start], out = [];
    try {
      let { logits } = await this.prefill(ids, side);
      for (let k = 0; k < n; k++) {
        const tok = sampleToken(logits, {}, [], suppress);
        out.push(tok);
        if (k === n - 1 || (stopAtEos && (tok === T.eos || tok === T.text_end))) break;
        ({ logits } = await this.decoder.step([tok]));
      }
    } finally { this.decoder.reset(); }
    return out;
  }

  // { content, reasoning }: the answer after </think> (special tokens skipped), and what the think block held
  decodeText(tokens) {
    const T = this.tokens, k = T.think_end == null ? -1 : tokens.indexOf(T.think_end);
    const dec = (t) => this.tokenizer.decode(t, { skip_special_tokens: true });
    if (k < 0) return { content: dec(tokens), reasoning: '' };
    return { content: dec(tokens.slice(k + 1)).trimStart(), reasoning: dec(tokens.slice(0, k)).replace(/^\s*<think>/, '').trim() };
  }

  async dispose() { this.ready = false; await this.decoder.dispose(); await this.vision.release(); await this.audio.release(); }
}

function concat(parts) {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
