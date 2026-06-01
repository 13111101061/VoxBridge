# VoxBridge

Zero-dependency multi-model TTS inference gateway. Protocol-driven adapter architecture with auto GPU detection. First adapter: **Qwen3 1.7B TTS** — full pipeline working on Vulkan + DirectML.

```
  text → tokenizer → talker (Vulkan GGUF) → predictor → ONNX decoder → WAV
```

## Quickstart

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

## Architecture

| Layer | Module | Description |
|---|---|---|
| Protocol | `runtime/adapter/protocol.py` | `TTSAdapter` interface: `load()` / `stream()` / `shutdown()` |
| Device | `runtime/device/device.py` | Auto-detect Vulkan/DirectML/CUDA/CPU |
| Registry | `runtime/registry/registry.py` | Model registry: register / get / list |
| Adapter | `adapters/qwen3/adapter.py` | Qwen3 1.7B full-streaming pipeline |

## Performance (RX 6900XT)

| Stage | Latency | Backend |
|---|---|---|
| Talker / frame | ~14ms | Vulkan GGUF |
| Predictor / frame | ~6ms | Vulkan GGUF |
| ONNX Decoder / chunk | ~50ms | DirectML |
| Total "hello" (10 frames) | ~0.5s | — |

~18x faster than CPU inference.

## Setup

```bash
pip install numpy onnxruntime tokenizers soundfile scipy
# Place GGUF / ONNX / NPY model files under models/ (see README_CN.md)
python probes/run_e2e.py
```

## Roadmap

- [x] Embedding batch injection via ctypes
- [x] Vulkan GPU acceleration
- [x] E2E WAV output (Vulkan + DirectML)
- [x] Speaker encoder + Mel extraction for voice cloning
- [ ] Streaming pipeline (threading/queue)
- [ ] WebSocket API Gateway
- [ ] Multi-model support (CosyVoice, ChatTTS)

## Docs

[README_CN.md](README_CN.md) — 中文文档 (architecture, setup, voice cloning, tech details)

## License

MIT
