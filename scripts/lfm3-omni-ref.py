#!/usr/bin/env python3
"""lfm3-omni-ref.py: the CPU reference fixture for engines/lfm3_omni.js (LiquidAI LFM3-Omni-2B, ONNX).

Runs Liquid's own runner (Liquid4All/onnx-export, liquidonnx.lfm3_omni.session.ONNXOmniModel) on onnxruntime's CPU EP
with the graphs the browser engine loads: decoder_q4f16, vision_encoder_q8 and the fp32 audio_encoder. Writes
test/refs/lfm3-omni/: fixture.json (per case the prompt ids, the first 32 greedy tokens and their text, the mel and
pixel_values checksums) plus the inputs as raw little-endian blobs, so the Node tests and the browser gate need no
image or WAV decoder:

  image.rgb       uint8 H × W × 3 (a scene drawn here with PIL: no third-party image)
  speech.pcm16    int16 mono 16 kHz (macOS `say` + `afconvert`, made here when missing)
  speech.mel.f32  float32 [128, T]: log_mel of speech.pcm16 / 32768, the full input_features

  cd <onnx-export checkout> && uv run python /path/to/diskformer.js/scripts/lfm3-omni-ref.py \
      --model /Volumes/Models2/Models/lfm3/omni-2b-onnx [--out test/refs/lfm3-omni]

Cases (all chat mode as lfm3-omni-infer runs it: <|mixed_start|> and <|audio_start|> suppressed, greedy, thinking
off, which is the chat template's default): text, image + text, speech only. The fixture's `speech` notes are what
the later speech-output layer (depth decoder, codec, iSTFT) needs; nothing here runs those graphs.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from liquidonnx.lfm3_audio.session import TextSampling, log_mel
from liquidonnx.lfm3_omni.session import ONNXOmniModel
from liquidonnx.lfm3_vl.session import expand_image_tokens, preprocess_images

GEN = 32
SPEECH_TEXT = 'Hello there. Could you tell me which mountain is the tallest in the world, and how high it is?'
CASES = [
    {'name': 'text', 'content': [{'type': 'text', 'text': 'What is the capital of France? Answer in one sentence.'}]},
    {'name': 'image', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Describe this image in one sentence.'}]},
    {'name': 'speech', 'content': [{'type': 'audio'}]},
]


class RefModel(ONNXOmniModel):
    # the browser's graphs: q4f16 decoder, int8 vision, fp32 speech encoder; no speech-out graphs (layer 1)
    bundles = {'web': {'decoder': 'decoder_q4f16.onnx', 'audio_encoder': 'audio_encoder.onnx',
                       'vision_encoder': 'vision_encoder_q8.onnx'}}


def draw_image(width=333, height=250) -> Image.Image:
    """A sky, a sun, hills and a house. 333 × 250 is resized to 320 × 256 (one side down, one up)."""
    im = Image.new('RGB', (width, height), (110, 170, 235))
    d = ImageDraw.Draw(im)
    d.ellipse((240, 20, 300, 80), fill=(255, 215, 40))
    d.polygon([(0, 190), (90, 130), (200, 185), (333, 120), (333, 250), (0, 250)], fill=(60, 150, 60))
    d.rectangle((70, 150, 150, 215), fill=(200, 60, 50))
    d.polygon([(60, 150), (110, 105), (160, 150)], fill=(120, 60, 30))
    d.rectangle((100, 180, 120, 215), fill=(80, 40, 20))
    return im


def make_speech(path: pathlib.Path):
    with tempfile.TemporaryDirectory() as tmp:
        aiff = pathlib.Path(tmp) / 'speech.aiff'
        subprocess.run(['say', '-o', str(aiff), SPEECH_TEXT], check=True)
        wav = pathlib.Path(tmp) / 'speech.wav'
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', str(aiff), str(wav)], check=True)
        import scipy.io.wavfile
        rate, samples = scipy.io.wavfile.read(str(wav))
        assert rate == 16000 and samples.dtype == np.int16, (rate, samples.dtype)
        path.write_bytes(samples.astype('<i2').tobytes())


def checksum(a: np.ndarray, every: int) -> dict:
    """sum, sum of squares, an index-weighted sum and every `every`-th value (flat order), in float64."""
    f = a.astype(np.float64).reshape(-1)
    idx = np.arange(f.size, dtype=np.float64)
    return {'shape': list(a.shape), 'sum': float(f.sum()), 'sumsq': float((f * f).sum()),
            'wsum': float(((idx % 1009) * f).sum()), 'every': every, 'samples': [float(x) for x in f[::every]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=str(pathlib.Path(__file__).resolve().parent.parent / 'test/refs/lfm3-omni'))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = RefModel(args.model, 'web', force_cpu=True, tf32=False)
    model.load()
    cfg, tok = model.config, model.config.tokens
    merge = model.vision_config.merge_size

    image = draw_image()
    rgb = np.asarray(image, dtype=np.uint8)
    (out / 'image.rgb').write_bytes(rgb.tobytes())
    pcm_path = out / 'speech.pcm16'
    if not pcm_path.exists():
        make_speech(pcm_path)
    wave = (np.frombuffer(pcm_path.read_bytes(), dtype='<i2').astype(np.float32) / 32768.0)
    mel = log_mel(wave, cfg.mel, model.mel_filters)
    (out / 'speech.mel.f32').write_bytes(mel.astype('<f4').tobytes())

    pixel_values, grids = preprocess_images(model.image_processor, [image])
    image_features = None
    results = []
    for case in CASES:
        messages = [{'role': 'user', 'content': case['content']}]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        ids = model.tokenizer(prompt, add_special_tokens=False)['input_ids']
        has_image = any(c['type'] == 'image' for c in case['content'])
        has_audio = any(c['type'] == 'audio' for c in case['content'])
        audio_count = cfg.mel.num_tokens(mel.shape[1]) if has_audio else 0
        ids = [t for x in ids for t in ([x] * audio_count if x == tok.audio else [x])]
        ids = expand_image_tokens(ids, grids if has_image else np.zeros((0, 3), np.int64), model.vision_tokens.image,
                                  merge)
        audio_features = model.encode_audio([wave]) if has_audio else None
        if has_image and image_features is None:
            image_features, _ = model.encode_images([image])
        gen = model.generate_ids(ids, audio_features, max_new_tokens=GEN, text_sampling=TextSampling(),
                                 suppress_mixed=True, suppress_audio_out=True,
                                 image_features=image_features if has_image else None)
        row = {'name': case['name'], 'messages': messages, 'enable_thinking': False, 'prompt': prompt,
               'prompt_ids': ids, 'gen': gen.tokens, 'text': model.tokenizer.decode(gen.tokens)}
        if has_image:
            row['image_tokens'] = int(np.prod(grids[0])) // merge ** 2
            row['image_features'] = checksum(image_features, 4099)
        if has_audio:
            row['audio_tokens'] = audio_count
            row['audio_features'] = checksum(audio_features, 4099)
        results.append(row)
        print(case['name'], len(ids), 'prompt ids;', repr(row['text']))

    raw = json.loads((pathlib.Path(args.model) / 'config.json').read_text())
    fixture = {
        'model': 'LiquidAI/LFM3-Omni-2B-ONNX',
        'graphs': model.bundles['web'],
        'runner': 'liquidonnx (Liquid4All/onnx-export) ONNXOmniModel, onnxruntime CPU EP',
        'image': {'file': 'image.rgb', 'width': image.width, 'height': image.height, 'grid': grids[0].tolist(),
                  'pixel_values': checksum(pixel_values, 97)},
        'speech': {'file': 'speech.pcm16', 'text': SPEECH_TEXT, 'samples': int(wave.size), 'mel_file': 'speech.mel.f32',
                   'mel': checksum(mel, 101), 'audio_tokens': cfg.mel.num_tokens(mel.shape[1])},
        'tokens': {'audio': tok.audio, 'audio_out': tok.audio_out, 'audio_start': tok.audio_start,
                   'text_end': tok.text_end, 'mixed_start': tok.mixed_start, 'eos': tok.eos, 'bos': tok.bos,
                   'image': raw['image_token_id'], 'video': raw['video_token_id']},
        # what the speech-output layer will need (not built yet): lfm3_audio.session.sample_frame / generate_ids
        'speech_out': {'num_codebooks': cfg.num_codebooks, 'codebook_size': cfg.codebook_size,
                       'audio_eos_code': cfg.audio_eos_code, 'maskgit_schedule': cfg.maskgit_schedule,
                       'interleaved_n_text': cfg.interleaved_n_text, 'interleaved_n_audio': cfg.interleaved_n_audio,
                       'codec': {'fsq_levels': cfg.fsq_levels, 'quantizer_dim': cfg.codec_quantizer_dim,
                                 'frame_stack': cfg.codec_frame_stack, 'n_fft': cfg.codec_n_fft,
                                 'hop_length': cfg.codec_hop_length, 'sampling_rate': cfg.codec_sampling_rate,
                                 'frame_samples': cfg.frame_samples},
                       'graphs': {'depth_decoder': 'depth_decoder.onnx (fp32 in every bundle)',
                                  'codec_decoder': 'codec_decoder.onnx'},
                       'generation_config': json.loads((pathlib.Path(args.model) / 'generation_config.json').read_text())},
        'files_sha256': {f: hashlib.sha256((out / f).read_bytes()).hexdigest()
                         for f in ('image.rgb', 'speech.pcm16', 'speech.mel.f32')},
        'cases': results,
    }
    (out / 'fixture.json').write_text(json.dumps(fixture, indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
