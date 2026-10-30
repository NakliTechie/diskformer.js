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
 * Speech output (output_modalities: ['text', 'audio']) is lfm3-omni-infer's s2s mode, lfm3_audio.session.generate_ids'
 * TEXT / AUDIO / INTERLEAVED machine: the system turn "Respond with interleaved text and audio." (when the messages
 * have none) and <|mixed_start|> after the generation prompt; then 6 text slots and 12 speech frames in turn until
 * <|text_end|>, after which speech goes on to its end-of-audio frame. A frame is 12 codes from the fp32 depth
 * decoder, MaskGIT in 3 passes of 4 codebooks over the decoder's hidden state (audio_temperature 0.4, repetition
 * penalty 1.2 over the last 64 frames, from generation_config.json); it goes back into the decoder as <|audio_out|>
 * with audio_out_codes. Every 12 frames (and at the end) the fp32 codec decoder turns the new frames, with 20 frames
 * of left context, into STFT rows, and an overlap-add inverse STFT gives 24 kHz Float32 PCM to onAudio.
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
  // speech out, loaded with load({ speech: true }) or on the first generate() that asks for audio
  speechGraphs: { depth: 'onnx/depth_decoder.onnx', codec: 'onnx/codec_decoder.onnx' },
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

// ── Speech out: MaskGIT frames, the codec, the inverse STFT ───────────────────────────────────────────────────────
// Codec frames re-decoded before each chunk's new frames. The codec is causal (5 conv layers of kernel 7, 3 attention
// layers of window 30, over 6 rows per frame), so a row sees at most 5·6 + 3·29 = 117 rows back: 20 frames covers it.
export const CODEC_CONTEXT_FRAMES = 20;
export const SPEECH_SYSTEM_PROMPT = 'Respond with interleaved text and audio.';

// A random permutation of 0..n-1 (Fisher-Yates over rand): one MaskGIT order
export function randomOrder(n, rand = Math.random) {
  const o = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [o[i], o[j]] = [o[j], o[i]]; }
  return o;
}

// Codes [12] of one frame (lfm3_audio.session.sample_frame): the order split by the schedule; per pass one depth run
// on (hidden, codes so far, mask of the hidden codebooks) and each picked codebook sampled from its logits with the
// repetition penalty over its own column of history; all audio_eos_code when any code is.
// depth(hidden, codes BigInt64Array [12], masked Uint8Array [12]) → Float32Array [12 · codebook_size]
export async function sampleFrame(hidden, depth, { order, schedule = [4, 4, 4], codebook_size = 2017, audio_eos_code = 2016, sampling = {}, history = [], rand = Math.random }) {
  const K = order.length;
  if ([...order].sort((a, b) => a - b).some((v, i) => v !== i)) throw new Error(`MaskGIT order ${order} is not a permutation`);
  const codes = new BigInt64Array(K), masked = new Uint8Array(K).fill(1);
  let at = 0;
  for (const n of schedule) {
    const pick = order.slice(at, at + n); at += n;
    const logits = await depth(hidden, codes, masked);
    for (const k of pick) {
      const col = history.map((f) => f[k]);
      codes[k] = BigInt(sampleToken(logits.subarray(k * codebook_size, (k + 1) * codebook_size), sampling, col, [], rand));
    }
    for (const k of pick) masked[k] = 0;
  }
  const frame = Array.from(codes, Number);
  return frame.includes(audio_eos_code) ? frame.fill(audio_eos_code) : frame;
}

// Speech sampling of generate()'s audio_* options, as lfm3_audio's sample_token reads them: temperature 0 (or none
// and no top-k) is argmax; a top-k without a temperature samples at temperature 1.
export function audioSampling({ audio_temperature, audio_top_k, audio_repetition_penalty } = {}) {
  const greedy = audio_temperature === 0 || (audio_temperature == null && !audio_top_k);
  return { do_sample: !greedy, temperature: audio_temperature ?? 1, top_k: audio_top_k ?? 0, repetition_penalty: audio_repetition_penalty ?? 1 };
}

// liquid-audio's inverse STFT with "same" padding (lfm2_audio.infer.istft_same_padding), fed row by row: each row is
// codec_decoder.onnx's [log-magnitude (n_fft/2+1), phase (n_fft/2+1)]; irfft, periodic Hann (float64), overlap-add,
// divide by the summed window², trim (n_fft − hop)/2 at both ends. push() returns the samples no later row can change;
// end() the rest. Offline and streamed output are the same samples.
export class Istft {
  constructor(n_fft = 1280, hop = 320) {
    this.N = n_fft; this.hop = hop; this.bins = n_fft / 2 + 1; this.pad = (n_fft - hop) >> 1;
    this.win = Float64Array.from({ length: n_fft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n_fft));
    this.cos = Float64Array.from({ length: n_fft }, (_, i) => Math.cos((2 * Math.PI * i) / n_fft));
    this.sin = Float64Array.from({ length: n_fft }, (_, i) => Math.sin((2 * Math.PI * i) / n_fft));
    this.wave = new Float64Array(0); this.env = new Float64Array(0); this.rows = 0; this.emitted = 0;
  }
  #grow(size) {
    if (this.wave.length >= size) return;
    const n = Math.max(size, this.wave.length * 2), w = new Float64Array(n), e = new Float64Array(n);
    w.set(this.wave); e.set(this.env); this.wave = w; this.env = e;
  }
  // rows: Float32Array of whole rows (n_fft + 2 each)
  push(rows) {
    const { N, bins, hop } = this, width = 2 * bins, count = rows.length / width;
    this.#grow((this.rows + count - 1) * hop + N);
    const { re, im, xr, xi, tmp } = (this.buf ??= { re: new Float64Array(N), im: new Float64Array(N), xr: new Float64Array(N), xi: new Float64Array(N), tmp: new Float64Array(2 * N) });
    for (let r = 0; r < count; r++) {
      const base = r * width, o = (this.rows + r) * hop;
      for (let k = 0; k < bins; k++) { const m = Math.exp(rows[base + k]), ph = rows[base + bins + k]; re[k] = m * Math.cos(ph); im[k] = m * Math.sin(ph); }
      // numpy irfft: the imaginary parts of bins 0 and N/2 are dropped; the rest of the spectrum is the conjugate mirror
      im[0] = 0; im[bins - 1] = 0;
      for (let k = bins; k < N; k++) { re[k] = re[N - k]; im[k] = -im[N - k]; }
      this.#ifft(re, im, 0, 1, N, xr, xi, 0, tmp);
      for (let n = 0; n < N; n++) {
        this.wave[o + n] += (xr[n] / N) * this.win[n];
        this.env[o + n] += this.win[n] * this.win[n];
      }
    }
    this.rows += count;
    return this.#take(this.rows * hop);
  }
  // Unnormalised inverse DFT (exp(+2πi·kn/N)) of the n points of (re, im) from `off` every `stride`, into (xr, xi)
  // from `at`: mixed-radix Cooley-Tukey (n_fft 1280 = 2⁸·5), each level a naive DFT of its smallest prime factor.
  #ifft(re, im, off, stride, n, xr, xi, at, tmp) {
    if (n === 1) { xr[at] = re[off]; xi[at] = im[off]; return; }
    let p = 2; while (n % p) p++;
    const m = n / p, N = this.N, step = N / n;
    for (let r = 0; r < p; r++) this.#ifft(re, im, off + r * stride, stride * p, m, xr, xi, at + r * m, tmp);
    for (let k = 0; k < m; k++) {
      for (let r = 0; r < p; r++) { tmp[2 * r] = xr[at + r * m + k]; tmp[2 * r + 1] = xi[at + r * m + k]; }
      for (let q = 0; q < p; q++) {
        const j = k + q * m;
        let sr = tmp[0], si = tmp[1];
        for (let r = 1; r < p; r++) {
          const a = ((r * j) % n) * step, c = this.cos[a], s = this.sin[a], yr = tmp[2 * r], yi = tmp[2 * r + 1];
          sr += yr * c - yi * s; si += yr * s + yi * c;
        }
        xr[at + j] = sr; xi[at + j] = si;
      }
    }
  }
  end() { return this.rows ? this.#take((this.rows - 1) * this.hop + this.N - this.pad) : new Float32Array(0); }
  #take(upTo) {
    const from = this.pad + this.emitted, to = Math.max(from, upTo);
    const out = new Float32Array(to - from);
    for (let i = 0; i < out.length; i++) out[i] = this.wave[from + i] / this.env[from + i];
    this.emitted += out.length;
    return out;
  }
}

// Streams one utterance's frames to PCM: decode(final) runs the codec on the frames not yet decoded plus up to
// CODEC_CONTEXT_FRAMES before them, keeps the new frames' rows and pushes them through the inverse STFT.
// codec(codes BigInt64Array [12 · F], F) → Float32Array [6F · (n_fft + 2)]
export class SpeechStream {
  constructor(codec, { num_codebooks = 12, frame_stack = 6, n_fft = 1280, hop = 320, context = CODEC_CONTEXT_FRAMES } = {}) {
    Object.assign(this, { codec, K: num_codebooks, stack: frame_stack, width: n_fft + 2, context });
    this.istft = new Istft(n_fft, hop); this.frames = []; this.done = 0;
  }
  push(frame) { this.frames.push(frame); }
  get pending() { return this.frames.length - this.done; }
  async decode(final = false) {
    const parts = [];
    if (this.pending) {
      const start = Math.max(0, this.done - this.context), F = this.frames.length - start, K = this.K;
      // audio_codes [1, 12, F]: codebook-major
      const codes = new BigInt64Array(K * F);
      for (let f = 0; f < F; f++) for (let k = 0; k < K; k++) codes[k * F + f] = BigInt(this.frames[start + f][k]);
      const rows = await this.codec(codes, F);
      parts.push(this.istft.push(rows.subarray((this.done - start) * this.stack * this.width)));
      this.done = this.frames.length;
    }
    if (final) parts.push(this.istft.end());
    return concat(parts);
  }
}

// ── The engine ────────────────────────────────────────────────────────────────────────────────────────────────────
export class Lfm3Omni {
  // opts: baseUrl (the repo layout) or files { 'onnx/decoder_q4f16.onnx': URL | Blob | ArrayBuffer | Uint8Array, … };
  // ort (onnxruntime-web/webgpu module), transformers (for PreTrainedTokenizer) or tokenizer; device 'webgpu';
  // onProgress({ status: 'download' | 'cache', file, loaded, total }) while a file arrives, then 'session' per graph;
  // cacheName (CACHE_NAME; null: no Cache Storage) and force (download again, replacing the stored copies);
  // speech: also load the speech-out graphs now (else the first generate() with audio output loads them)
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
      const file = LFM3_OMNI_2B.graphs[role] ?? LFM3_OMNI_2B.speechGraphs[role];
      const model = await bytes(file), data = await bytes(file + '_data');
      onProgress({ status: 'session', file });
      return { model, data, dataPath: file.split('/').pop() + '_data' };
    };
    const sessionOpts = { device };
    const vision = await createSession(ort, await graph('vision'), sessionOpts);
    const audio = await createSession(ort, await graph('audio'), sessionOpts);
    const decoder = await OrtLfm3Decoder.create(ort, await graph('decoder'), sessionOpts, config.text_config);
    const m = new Lfm3Omni({ ort, tokenizer, config, gen, processor, vision, audio, decoder });
    // the speech graphs, built when first needed from the same file source and cache
    m.loadSpeechGraphs = async () => ({ depth: await createSession(ort, await graph('depth'), sessionOpts), codec: await createSession(ort, await graph('codec'), sessionOpts) });
    if (opts.speech) await m.loadSpeech();
    onProgress({ status: 'ready' });
    return m;
  }

  // The depth and codec sessions (speech out); loaded once
  async loadSpeech() {
    if (!this.depth) {
      if (!this.loadSpeechGraphs) throw new Error('no speech graphs: load the engine with Lfm3Omni.load()');
      this.speechLoading ??= this.loadSpeechGraphs().then(({ depth, codec }) => { this.depth = depth; this.codec = codec; });
      await this.speechLoading;
    }
    return this;
  }
  // depth_decoder.onnx: logits [12 · codebook_size] of one frame
  async depthLogits(hidden, codes, masked) {
    const T = this.ort.Tensor, K = codes.length;
    const out = await this.depth.run({ hidden_states: new T('float32', hidden, [1, hidden.length]), codes: new T('int64', codes, [1, K]), masked: new T('bool', masked, [1, K]) });
    const l = Float32Array.from(out.logits.data); out.logits.dispose?.();
    return l;
  }
  // codec_decoder.onnx: stft_features [6F · (n_fft + 2)] of frames (codebook-major codes [12 · F])
  async codecRows(codes, F) {
    const out = await this.codec.run({ audio_codes: new this.ort.Tensor('int64', codes, [1, this.speech.num_codebooks, F]) });
    const r = Float32Array.from(out.stft_features.data); out.stft_features.dispose?.();
    return r;
  }

  // Every file load() reads, relative to baseUrl (with speech: also the speech-out graphs)
  static files({ speech = false } = {}) {
    const graphs = { ...LFM3_OMNI_2B.graphs, ...(speech ? LFM3_OMNI_2B.speechGraphs : {}) };
    return [...LFM3_OMNI_2B.json, 'chat_template.jinja', ...Object.values(graphs).flatMap((g) => [g, g + '_data'])];
  }
  // Stored copies of this model's files under baseUrl: [{ file, url, bytes }] (bytes from the stored length)
  static async cached({ baseUrl = LFM3_OMNI_2B.baseUrl, cacheName = CACHE_NAME } = {}) {
    if (!globalThis.caches) return [];
    const cache = await caches.open(cacheName), out = [];
    for (const file of Lfm3Omni.files({ speech: true })) {
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
    // speech out (lfm3_audio.config), and the codec's STFT (audio_tokenizer/config.json's values for this checkpoint)
    this.speech = { num_codebooks: t('num_codebooks', 12), codebook_size: t('codebook_size', 2017), audio_eos_code: t('audio_eos_code', 2016),
      maskgit_schedule: t('maskgit_schedule', [4, 4, 4]), interleaved_n_text: t('interleaved_n_text', 6), interleaved_n_audio: t('interleaved_n_audio', 12),
      frame_stack: 6, n_fft: 1280, hop: 320, sampling_rate: 24000 };
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
  // One reply at a time: generate() and greedyIds() share the decoder's caches and onnxruntime's sessions, and two runs
  // overlapping on WebGPU fail ("Failed to download data from buffer") or hang (2026-10-30: a caller that stopped
  // reading a text reply without aborting it, then asked for speech). A call made while another runs waits its turn.
  #turn = Promise.resolve();
  #serial(fn) { const run = this.#turn.then(fn); this.#turn = run.catch(() => {}); return run; }

  generate(messages, opts = {}) { return this.#serial(() => this.#generate(messages, opts)); }

  async #generate(messages, opts) {
    const { max_new_tokens = 1024, signal, onText, onToken, streamer, seed, output_modalities = ['text'] } = opts;
    if (output_modalities.includes('audio')) {
      if (!output_modalities.includes('text')) throw new Error("LFM3-Omni speaks in interleaved mode: output_modalities ['text', 'audio']");
      return this.#generateSpeech(messages, opts);
    }
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

  // Interleaved text and speech (lfm3_audio.session.generate_ids from <|mixed_start|>). max_new_tokens counts slots,
  // text tokens and frames alike. Text streams as in generate(); speech as onAudio(Float32Array 24 kHz) every
  // audio_chunk_frames frames and at the end; signal is read before every slot. orders: a function returning one
  // MaskGIT order per frame, or an array of them (a reference replay); default random from seed.
  // Resolves { text, stopReason, tokens, reasoning, content, audio: Float32Array, audio_codes: frames, sampling_rate }.
  async #generateSpeech(messages, opts) {
    const { max_new_tokens = 1024, signal, onText, onToken, onAudio, streamer, seed, audio_chunk_frames = 12 } = opts;
    await this.loadSpeech();
    const g = this.generationConfig, S = this.speech, T = this.tokens;
    const text = { do_sample: opts.do_sample ?? false, temperature: opts.temperature ?? 1, top_k: opts.top_k ?? 0,
      top_p: opts.top_p ?? 1, min_p: opts.min_p ?? 0, repetition_penalty: opts.repetition_penalty ?? 1 };
    const audio = audioSampling({ audio_temperature: opts.audio_temperature ?? g.audio_temperature ?? 0.4, audio_top_k: opts.audio_top_k ?? g.audio_top_k,
      audio_repetition_penalty: opts.audio_repetition_penalty ?? g.audio_repetition_penalty ?? 1.2 });
    const window = opts.audio_repetition_window ?? g.audio_repetition_window ?? 64;
    const rand = seed === undefined ? Math.random : mulberry32(seed);
    const orderOf = Array.isArray(opts.orders) ? ((i) => opts.orders[i]) : typeof opts.orders === 'function' ? opts.orders : () => randomOrder(S.num_codebooks, rand);
    const msgs = messages.some((m) => m.role === 'system') ? messages : [{ role: 'system', content: SPEECH_SYSTEM_PROMPT }, ...messages];
    const p = await this.prepare(msgs, { ...opts, enable_thinking: false });
    const ids = [...p.ids, T.mixed_start];
    const side = { image_features: p.pics.length ? await this.encodeImages(p.pics) : undefined, audio_features: p.mels.length ? await this.encodeAudio(p.mels) : undefined };
    const suppress = [T.audio, T.audio_out, T.image, T.video];
    const stream = new SpeechStream((c, F) => this.codecRows(c, F), { num_codebooks: S.num_codebooks, frame_stack: S.frame_stack, n_fft: S.n_fft, hop: S.hop });
    const tokens = [], textIds = [], textHistory = [], history = [], frames = [], chunks = [];
    let stopReason = 'length', sent = 0, mode = 'interleaved', slotsLeft = S.interleaved_n_text, speaking = false, textDone = false, nFrames = 0;
    const emitText = () => {
      if (!onText) return;
      const full = this.tokenizer.decode(textIds, { skip_special_tokens: true });
      if (full.length > sent && !full.endsWith('\uFFFD')) { onText(full.slice(sent)); sent = full.length; }
    };
    const flush = async (final) => { const pcm = await stream.decode(final); if (pcm.length) { chunks.push(pcm); onAudio?.(pcm); } };
    try {
      let { logits, hidden } = await this.prefill(ids, side);
      for (let step = 0; step < max_new_tokens; step++) {
        if (signal?.aborted) { stopReason = 'aborted'; break; }
        const interleaving = mode === 'interleaved';
        let next, codes, finished = false;
        if (mode === 'audio' || (interleaving && speaking)) {
          const frame = await sampleFrame(hidden, (h, c, m) => this.depthLogits(h, c, m), { order: orderOf(nFrames++), schedule: S.maskgit_schedule,
            codebook_size: S.codebook_size, audio_eos_code: S.audio_eos_code, sampling: audio, history, rand });
          if (frame[0] === S.audio_eos_code) mode = 'text';
          else {
            history.push(frame); if (history.length > window) history.shift();
            frames.push(frame); tokens.push(T.audio_out); stream.push(frame);
            if (stream.pending >= audio_chunk_frames) await flush(false);
          }
          next = T.audio_out; codes = BigInt64Array.from(frame, BigInt);
        } else {
          const tok = sampleToken(logits, text, textHistory, suppress, rand);
          tokens.push(tok); textHistory.push(tok); if (textHistory.length > 32) textHistory.shift();
          if (tok === T.text_end) { if (interleaving) textDone = speaking = true; else finished = true; }
          else if (tok === T.eos) finished = true;
          else if (!interleaving && tok === T.audio_start) mode = 'audio';
          else if (!interleaving && tok === T.mixed_start) { mode = 'interleaved'; slotsLeft = S.interleaved_n_text; speaking = false; textDone = false; }
          if (tok !== T.text_end && tok !== T.eos) { textIds.push(tok); streamer?.put?.([[BigInt(tok)]]); onToken?.(tok); emitText(); }
          next = tok;
        }
        if (interleaving && !textDone && --slotsLeft === 0) { speaking = !speaking; slotsLeft = speaking ? S.interleaved_n_audio : S.interleaved_n_text; }
        if (finished) { stopReason = 'eos'; break; }
        if (step === max_new_tokens - 1) break;
        ({ logits, hidden } = await this.decoder.step([next], codes ? { audio_out_codes: codes } : {}));
      }
      if (stopReason !== 'aborted') await flush(true);
    } finally { this.decoder.reset(); streamer?.end?.(); }
    const all = this.tokenizer.decode(textIds, { skip_special_tokens: true });
    if (onText && all.length > sent) onText(all.slice(sent));
    return { text: all, stopReason, tokens, ...this.decodeText(textIds), audio: concat(chunks), audio_codes: frames, sampling_rate: S.sampling_rate };
  }

  // Greedy ids after prompt ids (the gate's comparison): no template, no stop on EOS unless asked.
  greedyIds(ids, side = {}, n = 32, opts = {}) { return this.#serial(() => this.#greedyIds(ids, side, n, opts)); }

  async #greedyIds(ids, side = {}, n = 32, { stopAtEos = true } = {}) {
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

  async dispose() {
    this.ready = false;
    await this.decoder.dispose(); await this.vision.release(); await this.audio.release();
    if (this.depth) { await this.depth.release(); await this.codec.release(); this.depth = this.codec = null; }
  }
}

function concat(parts) {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
