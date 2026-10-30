#!/usr/bin/env python3
"""lfm3-omni-speech-ref.py: the CPU reference for engines/lfm3_omni.js's speech output (interleaved text + speech).

Runs Liquid's runner (Liquid4All/onnx-export, liquidonnx ONNXOmniModel.generate_ids and decode_audio) on onnxruntime's
CPU EP with the browser's graphs: decoder_q4f16, the fp32 audio_encoder, the fp32 depth_decoder and the fp32
codec_decoder (the q4 bundle's, as the model card pairs them on WebGPU). Writes test/refs/lfm3-omni/speech-out.json
and two small blobs:

  speech-out.json   the s2s case (lfm3-omni-infer --mode s2s: system "Respond with interleaved text and audio.",
                    the user's speech clip, assistant prefilled with <|mixed_start|>): prompt ids, every generated
                    slot, the speech frames' codes, the MaskGIT orders the runner drew (so the browser replays them),
                    and the decoded 24 kHz wave's length, RMS, peak and checksums
  istft.f32         codec_decoder.onnx's stft_features of the first ISTFT_FRAMES frames [6 F, 1282]
  istft.wave.f32    session.stft_to_wave of those rows (numpy irfft, periodic Hann, overlap-add, same padding)

Speech codes are greedy (audio temperature 0) with the checkpoint's repetition penalty 1.2 over 64 frames: the
runner's numpy sampling RNG cannot be replayed in a browser, its MaskGIT orders can (recorded here, seed 0).

  cd <onnx-export checkout> && uv run python /path/to/diskformer.js/scripts/lfm3-omni-speech-ref.py \
      --model /Volumes/Models2/Models/lfm3/omni-2b-onnx [--slots 72]
"""
import argparse
import json
import pathlib

import numpy as np

from liquidonnx.lfm3_audio.session import AudioSampling, TextSampling, log_mel, rng_orders, stft_to_wave
from liquidonnx.lfm3_omni.session import ONNXOmniModel

SYSTEM = 'Respond with interleaved text and audio.'
ISTFT_FRAMES = 4


class RefModel(ONNXOmniModel):
    bundles = {'web': {'decoder': 'decoder_q4f16.onnx', 'audio_encoder': 'audio_encoder.onnx',
                       'vision_encoder': 'vision_encoder_q8.onnx', 'depth_decoder': 'depth_decoder.onnx',
                       'codec_decoder': 'codec_decoder.onnx'}}


def stats(wave: np.ndarray) -> dict:
    w = wave.astype(np.float64)
    idx = np.arange(w.size, dtype=np.float64)
    return {'samples': int(w.size), 'rms': float(np.sqrt((w * w).mean())) if w.size else 0.0,
            'peak': float(np.abs(w).max()) if w.size else 0.0, 'sum': float(w.sum()),
            'wsum': float(((idx % 1009) * w).sum()), 'every': 997, 'values': [float(x) for x in w[::997]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--slots', type=int, default=72)
    ap.add_argument('--out', default=str(pathlib.Path(__file__).resolve().parent.parent / 'test/refs/lfm3-omni'))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    model = RefModel(args.model, 'web', force_cpu=True, tf32=False)
    model.load()
    cfg, tok = model.config, model.config.tokens

    base = json.loads((out / 'fixture.json').read_text())
    pcm = np.frombuffer((out / base['speech']['file']).read_bytes(), dtype='<i2')
    wave = pcm.astype(np.float32) / 32768.0
    mel = log_mel(wave, cfg.mel, model.mel_filters)
    count = cfg.mel.num_tokens(mel.shape[1])
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [{'type': 'audio'}]},
                {'role': 'assistant', 'content': '<|mixed_start|>'}]
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    ids = model.tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = [t for x in ids for t in ([x] * count if x == tok.audio else [x])]
    feats = model.encode_audio([wave])

    drawn, source = [], rng_orders(0, cfg.num_codebooks)

    def orders():
        o = source()
        drawn.append([int(x) for x in o])
        return o

    gen = model.generate_ids(ids, feats, max_new_tokens=args.slots, text_sampling=TextSampling(),
                             audio_sampling=AudioSampling(temperature=0.0, repetition_penalty=1.2, repetition_window=64),
                             orders=orders)
    codes = gen.audio_codes
    decoded = model.decode_audio(codes) if len(codes) else np.zeros(0, np.float32)
    text = [t for t in gen.tokens if t != tok.audio_out]

    # the iSTFT fixture: the codec's rows for the first frames and numpy's wave of them
    [rows] = model.codec_decoder.run(['stft_features'], {'audio_codes': codes[:ISTFT_FRAMES].T[None].astype(np.int64)})
    rows = rows[0].astype(np.float32)
    (out / 'istft.f32').write_bytes(rows.astype('<f4').tobytes())
    (out / 'istft.wave.f32').write_bytes(stft_to_wave(rows, cfg.codec_n_fft, cfg.codec_hop_length).astype('<f4').tobytes())

    fixture = {
        'model': 'LiquidAI/LFM3-Omni-2B-ONNX', 'graphs': model.bundles['web'],
        'runner': 'liquidonnx ONNXOmniModel.generate_ids + decode_audio, onnxruntime CPU EP',
        'mode': 's2s', 'system': SYSTEM, 'slots': args.slots, 'prompt': prompt, 'prompt_ids': ids,
        'audio_sampling': {'temperature': 0.0, 'repetition_penalty': 1.2, 'repetition_window': 64},
        'tokens': gen.tokens, 'text': model.tokenizer.decode(text, skip_special_tokens=True),
        'frames': codes.astype(int).tolist(), 'orders': drawn,
        'wave': {'sampling_rate': cfg.codec_sampling_rate, **stats(decoded)},
        'istft': {'frames': ISTFT_FRAMES, 'rows_file': 'istft.f32', 'rows': list(rows.shape), 'wave_file': 'istft.wave.f32',
                  'n_fft': cfg.codec_n_fft, 'hop_length': cfg.codec_hop_length},
    }
    (out / 'speech-out.json').write_text(json.dumps(fixture, indent=1) + '\n')
    print(len(ids), 'prompt ids;', len(codes), 'frames;', len(text), 'text tokens;', repr(fixture['text']))
    print('wave', fixture['wave']['samples'], 'samples, rms', round(fixture['wave']['rms'], 5))


if __name__ == '__main__':
    main()
