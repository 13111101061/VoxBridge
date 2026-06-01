[中文](README.md)

# VoxBridge — 零依赖多模型 TTS 推理网关

基于 StreamVox 生态的轻量 TTS 推理框架。运行时适配器协议 + 设备检测 + 模型注册，首个适配器 **Qwen3 1.7B TTS** 已跑通全链路。

```
  文本
    ↓ tokenizer (HuggingFace tokenizers, CPU)
  token IDs
    ↓ text_embedding_projected.npy 查表
  2048-dim 文本向量 + Speaker 嵌入
    ↓ talker.gguf (Vulkan GGUF, ~0.014s/decode)
  16 组 codec tokens × N 帧
    ↓ ONNX decoder (DirectML / CPU, stateful)
  24kHz WAV 音频流
```

## 目录结构

```
VoxBridge/
├── runtime/                    # 核心运行时（零外部依赖）
│   ├── adapter/protocol.py     # TTSAdapter 接口协议
│   ├── device/device.py        # GPU 后端检测 (Vulkan/CUDA/DirectML/CPU)
│   └── registry/registry.py    # 模型注册表
├── adapters/                   # 模型适配器实现
│   ├── qwen3/adapter.py        # Qwen3 1.7B TTS — 400行全链路推理
│   └── onnx_generic/           # 通用 ONNX 适配器（待实现）
├── config/                     # 配置文件（待填充）
├── gateway/                    # API 网关（待实现）
├── storage/                    # 提示词 / 缓存存储
└── probes/                     # 探针脚本 & 测试输出
    ├── run_e2e.py              # 一键端到端 GPU 推理测试
    ├── test_embd_c_api.py      # llama.cpp embedding 注入验证
    └── output/                 # 生成的 WAV / NPZ（被 .gitignore 排除）
```

## 架构分层

### 1. `TTSAdapter` 协议 (`runtime/adapter/protocol.py`)

所有适配器实现此接口：

```python
class TTSAdapter(Protocol):
    @property
    def info(self) -> ModelInfo: ...
    def load(self, *, device: str = "auto") -> None: ...
    def stream(self, text: str, prompt: PromptData | None = None, **kw) -> Iterator[np.ndarray]: ...
    def shutdown(self) -> None: ...
```

### 2. 设备检测 (`runtime/device/`)

`detect_device()` 自动选择最优后端：

| 平台 | 自动选择 |
|---|---|
| Windows (AMD GPU) | llama: Vulkan + ONNX: DirectML |
| Windows (NVIDIA GPU) | llama: CUDA + ONNX: CUDA |
| Linux | llama: Vulkan + ONNX: CPU |

支持 `device="gpu:0"`, `device="cpu"`, `device="auto"` 等参数。

### 3. 模型注册表 (`runtime/registry/`)

```python
from runtime.registry import ModelRegistry
from adapters.qwen3.adapter import Qwen3TTSAdapter

reg = ModelRegistry()
reg.register(Qwen3TTSAdapter("./models/qwen3-tts-clone-1.7b-gguf"))
adapter = reg.get("qwen3-tts-clone-1.7b-gguf")
adapter.load(device="auto")
for chunk in adapter.stream("你好世界"):
    play(chunk)
adapter.shutdown()
```

## 快速开始

### 安装

```bash
git clone https://github.com/13111101061/VoxBridge.git
cd VoxBridge
pip install numpy onnxruntime tokenizers soundfile scipy
# 从 https://github.com/batniel/StreamVox 安装 StreamVox
```

### 模型文件准备

将 Qwen3 1.7B TTS 模型放入 `models/qwen3-tts-clone-1.7b-gguf/` 目录：

```
qwen3-tts-clone-1.7b-gguf/
├── tokenizer.json                          # HuggingFace Tokenizer
├── qwen3_tts_decoder.fp16.onnx             # ONNX 音频解码器 (~350MB)
├── ckpt/clone_1.7B/
│   ├── qwen3_tts_talker.q5_k.gguf          # GGUF talker (~2.5GB)
│   ├── qwen3_tts_predictor.q8_0.gguf       # GGUF 预测器 (~700MB)
│   ├── qwen3_tts_speaker_encoder.fp16.onnx  # 音色编码器
│   └── embeddings/
│       ├── codec_embedding_0~15.npy        # Codec 嵌入表 (16个文件)
│       ├── text_embedding_projected.npy    # 文本→嵌入映射表 (~600MB)
│       ├── proj_weight.npy                 # 投影矩阵 (~32MB)
│       └── proj_bias.npy                   # 投影偏置
```

**注意**: 模型文件约 4GB，已加入 `.gitignore`，不要提交到 Git。

### 运行

```python
from adapters.qwen3.adapter import Qwen3TTSAdapter
import soundfile as sf, numpy as np

a = Qwen3TTSAdapter("./models/qwen3-tts-clone-1.7b-gguf")
a.load(device="auto")  # AMD: Vulkan+DirectML / NVIDIA: CUDA

chunks = list(a.stream("你好世界，欢迎使用VoxBridge。"))
audio = np.concatenate(chunks)
sf.write("output.wav", audio, 24000)
a.shutdown()
```

或直接运行探针脚本：

```bash
python probes/run_e2e.py
```

### 音色克隆

```python
import soundfile as sf
from runtime.adapter.protocol import PromptData

ref_audio, sr = sf.read("reference.wav", dtype='float32')
prompt = PromptData(model_name="qwen3-tts-clone-1.7b-gguf")
prompt.metadata["spk_audio"] = ref_audio

for chunk in a.stream("你好世界", prompt=prompt):
    play(chunk)  # 使用参考音频的音色
```

## 依赖

| 包 | 用途 |
|---|---|
| `numpy` | 向量计算 / 音频 / Mel |
| `onnxruntime` | ONNX 模型推理 |
| `tokenizers` (HuggingFace) | 文本 tokenize |
| `soundfile` | WAV 读写 |
| `scipy` | Mel 提取 (STFT) / 重采样 |
| `streamvox` | llama.cpp Python 绑定 (GGUF 推理) |

系统依赖：
- `ggml-vulkan.dll`（StreamVox 自带）
- `onnxruntime.dll`
- `DirectML.dll`（Windows GPU 解码）
- AMD: Vulkan Runtime | NVIDIA: CUDA Toolkit

## 性能参考 (RX 6900XT)

| 阶段 | 耗时 | 后端 |
|---|---|---|
| Tokenize | < 5ms | CPU |
| Talker / 帧 | ~14ms | Vulkan GGUF |
| Predictor / 帧 | ~6ms | Vulkan GGUF |
| ONNX Decoder / chunk | ~50ms | DirectML |
| **总计 "hello" (10帧)** | **~0.5s** | — |

CPU 推理约慢 18 倍（talker ~250ms/帧）。

## 技术要点 (Qwen3 适配器)

### Embedding 注入

llama.cpp Python 绑定原生不支持 embedding 输入。我们通过 `ctypes` 手动构造 `llama_batch` 结构体实现：

```python
batch = type(lm.llama_batch_get_one(...))()  # 获取结构体类型
batch.n_tokens = N
batch.token = NULL                           # 不传 token ID
batch.embd = embeddings.ctypes.data          # 注入向量
lm.llama_decode(ctx, batch)
```

### Vulkan 兼容性

- `llama_get_logits_ith()` 在 Vulkan 后端会挂起 — 改用 `llama_get_logits()`
- Vulkan 和 CPU 的 `llama_batch` 结构体布局不同 — 不可跨后端复用
- Talker GGUF 词表 3072（纯 codec token），EOS = token 0

### ONNX 解码器

- 输入 `audio_codes`: int64, shape `(batch, num_frames, 16)`
- 输出 `final_wav`: float16, shape `(batch, samples)`
- 有状态解码：跨 chunk 维护 8 层 KV cache、conv_history、latent_buffer、pre_conv_history

### Speaker Encoder

- ONNX 输入: `mels` float16 `(batch, T, 128)`
- Mel 参数: sr=24000, n_fft=1024, hop=256, n_mels=128, f_min=0, f_max=12000
- 输出: `spk_emb` float16 `(batch, 2048)`

## 开发路线

- [x] v0.1 运行时核 + Qwen3 适配器框架
- [x] v0.2 Embedding 批注入突破（ctypes 手动构造）
- [x] v0.3 Vulkan GPU 加速验证 (18x 提速)
- [x] v0.4 完整 GPU 管线 (talker + predictor + decoder)
- [x] v0.5 端到端 WAV 输出
- [x] v0.6 架构对齐官方 Qwen3-TTS 源码
- [x] v0.7 Speaker Encoder 集成
- [x] v0.8 真实 Mel 提取 → Speaker 嵌入 → 音色克隆
- [ ] v0.9 流式 pipeline（多线程/队列，实时 TTS）
- [ ] v0.10 Speaker Encoder ONNX GPU 加速
- [ ] v1.0 WebSocket API Gateway
- [ ] v1.1 Predictor KV cache 清理（修复长音频质量衰减）
- [ ] v1.2 Q8_0 / FP16 Talker 替换（提升音质）
- [ ] v2.0 多模型支持 (CosyVoice / ChatTTS)

## 已知问题

1. **音质**: talker 使用 Q5_K 量化，存在电子失真。建议替换为 Q8_0 或 FP16 GGUF。
2. **长音频**: Predictor KV cache 以位置 0..16 复写每帧，长音频可能累积失真。
3. **Windows 专用**: DirectML 和 Vulkan 后端目前仅在 Windows 验证通过。
4. **Speaker Encoder**: 参考音频需重采样至 24000Hz 单声道 float32。

## License

MIT
