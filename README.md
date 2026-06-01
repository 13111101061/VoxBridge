[中文文档](README_CN.md)

# VoxBridge — Zero-Dependency Multi-Model TTS Inference Gateway

A lightweight TTS inference framework built on the StreamVox ecosystem. Protocol-driven adapter architecture with auto GPU detection and model registry. The first adapter, **Qwen3 1.7B TTS**, runs the full pipeline end-to-end.

```
    text
      ↓ tokenizer (HuggingFace tokenizers, CPU)
    token IDs
      ↓ text_embedding_projected.npy lookup
    2048-dim text embeddings + Speaker embedding
      ↓ talker.gguf (Vulkan GGUF, ~0.014s/decode)
    16 codec tokens × N frames
      ↓ ONNX decoder (DirectML / CPU, stateful)
    24kHz WAV audio output
```

## Directory Structure

```
VoxBridge/
├── runtime/                    # Core runtime (zero external deps)
│   ├── adapter/protocol.py     # TTSAdapter interface protocol
│   ├── device/device.py        # GPU backend detection (Vulkan/CUDA/DirectML/CPU)
│   └── registry/registry.py    # Model registry
├── adapters/                   # Model adapter implementations
│   ├── qwen3/adapter.py        # Qwen3 1.7B TTS — 400-line full streaming pipeline
│   └── onnx_generic/           # Generic ONNX adapter (TBD)
├── config/                     # Configuration files (TBD)
├── gateway/                    # API gateway (TBD)
├── storage/                    # Prompt / cache storage
└── probes/                     # Probe scripts & test outputs
    ├── run_e2e.py              # One-click end-to-end GPU inference test
    ├── test_embd_c_api.py      # llama.cpp embedding injection verification
    └── output/                 # Generated WAV / NPZ (excluded by .gitignore)
```

## Architecture

### 1. `TTSAdapter` Protocol (`runtime/adapter/protocol.py`)

All adapters implement this interface:

```python
class TTSAdapter(Protocol):
    @property
    def info(self) -> ModelInfo: ...
    def load(self, *, device: str = "auto") -> None: ...
    def stream(self, text: str, prompt: PromptData | None = None, **kw) -> Iterator[np.ndarray]: ...
    def shutdown(self) -> None: ...
```

### 2. Device Detection (`runtime/device/`)

`detect_device()` automatically selects the optimal backend:

| Platform | Auto-Selected |
|---|---|
| Windows (AMD GPU) | llama: Vulkan + ONNX: DirectML |
| Windows (NVIDIA GPU) | llama: CUDA + ONNX: CUDA |
| Linux | llama: Vulkan + ONNX: CPU |

Accepts `device="gpu:0"`, `device="cpu"`, `device="auto"`, etc.

### 3. Model Registry (`runtime/registry/`)

```python
from runtime.registry import ModelRegistry
from adapters.qwen3.adapter import Qwen3TTSAdapter

reg = ModelRegistry()
reg.register(Qwen3TTSAdapter("./models/qwen3-tts-clone-1.7b-gguf"))
adapter = reg.get("qwen3-tts-clone-1.7b-gguf")
adapter.load(device="auto")
for chunk in adapter.stream("Hello world"):
    play(chunk)
adapter.shutdown()
```

## Quick Start

### Install

```bash
git clone https://github.com/13111101061/VoxBridge.git
cd VoxBridge
pip install numpy onnxruntime tokenizers soundfile scipy
# Install StreamVox from https://github.com/batniel/StreamVox
```

### Prepare Model Files

Place the Qwen3 1.7B TTS models under `models/qwen3-tts-clone-1.7b-gguf/`:

```
qwen3-tts-clone-1.7b-gguf/
├── tokenizer.json                          # HuggingFace Tokenizer
├── qwen3_tts_decoder.fp16.onnx             # ONNX audio decoder (~350MB)
├── ckpt/clone_1.7B/
│   ├── qwen3_tts_talker.q5_k.gguf          # GGUF talker (~2.5GB)
│   ├── qwen3_tts_predictor.q8_0.gguf       # GGUF predictor (~700MB)
│   ├── qwen3_tts_speaker_encoder.fp16.onnx  # Speaker encoder
│   └── embeddings/
│       ├── codec_embedding_0~15.npy        # Codec embedding tables (16 files)
│       ├── text_embedding_projected.npy    # Text→embedding lookup (~600MB)
│       ├── proj_weight.npy                 # Projection matrix (~32MB)
│       └── proj_bias.npy                   # Projection bias
```

**Important**: Model files are ~4GB total — do NOT commit to Git (already in `.gitignore`).

### Run

```python
from adapters.qwen3.adapter import Qwen3TTSAdapter
import soundfile as sf, numpy as np

a = Qwen3TTSAdapter("./models/qwen3-tts-clone-1.7b-gguf")
a.load(device="auto")  # Vulkan+DirectML on AMD, CUDA on NVIDIA

chunks = list(a.stream("Hello world, this is VoxBridge."))
audio = np.concatenate(chunks)
sf.write("output.wav", audio, 24000)
a.shutdown()
```

Or run the probe script:

```bash
python probes/run_e2e.py
```

### Voice Cloning

```python
import soundfile as sf
from runtime.adapter.protocol import PromptData

ref_audio, sr = sf.read("reference.wav", dtype='float32')
prompt = PromptData(model_name="qwen3-tts-clone-1.7b-gguf")
prompt.metadata["spk_audio"] = ref_audio

for chunk in a.stream("Hello world", prompt=prompt):
    play(chunk)  # Uses reference speaker identity
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Vector math / audio / Mel |
| `onnxruntime` | ONNX model inference |
| `tokenizers` (HuggingFace) | Text tokenization |
| `soundfile` | WAV I/O |
| `scipy` | Mel extraction (STFT) / resampling |
| `streamvox` | llama.cpp Python bindings (GGUF inference) |

System requirements:
- `ggml-vulkan.dll` (bundled with StreamVox)
- `onnxruntime.dll`
- `DirectML.dll` (Windows, for GPU decoding)
- AMD: Vulkan Runtime | NVIDIA: CUDA Toolkit

## Performance (RX 6900XT)

| Stage | Latency | Backend |
|---|---|---|
| Tokenize | < 5ms | CPU |
| Talker / frame | ~14ms | Vulkan GGUF |
| Predictor / frame | ~6ms | Vulkan GGUF |
| ONNX Decoder / chunk | ~50ms | DirectML |
| **Total "hello" (10 frames)** | **~0.5s** | — |

CPU inference is ~18x slower (talker ~250ms/frame).

## Technical Highlights (Qwen3 Adapter)

### Embedding Injection

llama.cpp Python bindings do not natively support embedding input mode. We manually construct the `llama_batch` struct via `ctypes`:

```python
batch = type(lm.llama_batch_get_one(...))()  # Get struct type
batch.n_tokens = N
batch.token = NULL                           # No token IDs
batch.embd = embeddings.ctypes.data          # Inject vectors instead
lm.llama_decode(ctx, batch)
```

### Vulkan Compatibility

- `llama_get_logits_ith()` hangs on Vulkan backend — use `llama_get_logits()` instead
- Vulkan and CPU `llama_batch` struct layouts differ — do NOT reuse across backends
- Talker GGUF vocab size = 3072 (codec tokens only), EOS = token 0

### ONNX Decoder

- Input `audio_codes`: int64 tensor, shape `(batch, num_frames, 16)`
- Output `final_wav`: float16, shape `(batch, samples)`
- Stateful decoding: maintain 8-layer KV cache, conv_history, latent_buffer, pre_conv_history across chunks

### Speaker Encoder

- ONNX input: `mels` float16 `(batch, T, 128)`
- Mel parameters: sr=24000, n_fft=1024, hop=256, n_mels=128, f_min=0, f_max=12000
- Output: `spk_emb` float16 `(batch, 2048)`

## Roadmap

- [x] v0.1 Runtime core + Qwen3 adapter framework
- [x] v0.2 Embedding batch injection via ctypes
- [x] v0.3 Vulkan GPU acceleration verified (18x speedup)
- [x] v0.4 Full GPU pipeline (talker + predictor + decoder)
- [x] v0.5 End-to-end WAV output
- [x] v0.6 Architecture alignment with official Qwen3-TTS
- [x] v0.7 Speaker encoder integration
- [x] v0.8 Real Mel extraction + Speaker embedding → voice cloning
- [ ] v0.9 Streaming pipeline (threading/queue, real-time TTS)
- [ ] v0.10 Speaker encoder ONNX with GPU acceleration
- [ ] v1.0 WebSocket API Gateway
- [ ] v1.1 Predictor KV cache clearing (fix long-audio quality degradation)
- [ ] v1.2 Q8_0 / FP16 talker replacement (audio quality improvement)
- [ ] v2.0 Multi-model support (CosyVoice, ChatTTS)

## Known Issues

1. **Audio quality**: Talker uses Q5_K quantization causing electronic distortion. Replace with Q8_0 or FP16 GGUF.
2. **Long audio**: Predictor KV cache reuses positions 0..16 per frame — may accumulate artifacts.
3. **Windows-only**: DirectML and Vulkan backends currently verified on Windows only.
4. **Speaker encoder**: Reference audio must be resampled to 24000Hz mono float32.

## License

MIT
