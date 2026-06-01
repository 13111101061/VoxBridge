"""Qwen3 TTS e2e test — Vulkan GPU + ONNX decoder → WAV"""
import sys; sys.path.insert(0, r'E:\D G\AI\AI-TTS')
import streamvox.models.qwen3.llama as lm; lm.init_llama_lib()
from adapters.qwen3.adapter import Qwen3TTSAdapter, MAX_NEW_TOKENS, _make_embd_batch
import numpy as np, time, soundfile as sf

import adapters.qwen3.adapter as ada
ada.MAX_NEW_TOKENS = 10

model_dir = r'E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf'
a = Qwen3TTSAdapter(model_dir); ckpt = a._ckpt

print('[1/5] Loading talker (Vulkan)...')
t0 = time.time()
a._talker_m = lm.load_model(str(ckpt / 'qwen3_tts_talker.q5_k.gguf'), n_gpu_layers=-1, backend='vulkan')
a._talker_ctx = lm.create_context(a._talker_m, n_ctx=512, n_batch=512, embeddings=True, no_perf=True)
print(f'    ({time.time()-t0:.1f}s)')

print('[2/5] Loading predictor (Vulkan)...')
t0 = time.time()
a._pred_m = lm.load_model(str(ckpt / 'qwen3_tts_predictor.q8_0.gguf'), n_gpu_layers=-1, backend='vulkan')
a._pred_ctx = lm.create_context(a._pred_m, n_ctx=512, n_batch=512, embeddings=True, no_perf=True)
print(f'    ({time.time()-t0:.1f}s)')

print('[3/5] Loading assets...')
a._load_assets()

print('[4/5] Loading ONNX decoder...')
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
a._decoder_sess = ort.InferenceSession(str(ckpt.parent / 'qwen3_tts_decoder.fp16.onnx'), sess_options=so, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
a._loaded = True

print('[5/5] Generating audio...')
t0 = time.time()
codes = a._invoke_talker(a._tokenize('hello'))
print(f'    Codes: {codes.shape} ({time.time()-t0:.1f}s)')

latent = a._project(codes)
print(f'    Latent: {latent.shape}')

chunks = list(a._decode_stream(latent))
audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
t1 = time.time()
print(f'    Audio: {len(audio)}smp, {len(audio)/24000:.1f}s, total {t1-t0:.1f}s')

out = r'E:\D G\AI\AI-TTS\probes\output\test_hello_gpu.wav'
sf.write(out, audio, 24000)
print(f'\nWAV saved: {out} ({len(audio)/24000:.1f}s)')
print('DONE!')
