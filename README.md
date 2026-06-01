[English](README_EN.md)

# VoxBridge — 零依赖多模型 TTS 推理网关

基于 StreamVox 生态的轻量 TTS 推理框架。运行时适配器协议 + 设备检测 + 模型注册，首个适配器 **Qwen3 1.7B TTS** 已跑通全链路。

```
    ↓ tokenizer (HuggingFace tokenizers, CPU)
  文本 token IDs
    ↓ text_embedding_projected.npy 查表
  2048-dim 文本向量 + Speaker 嵌入
    ↓ talker.gguf (Vulkan GGUF, ~0.014s/decode)
  16 组 codec tokens × N 帧
    ↓ ONNX decoder (DirectML / CPU, stateful)
```

## 目录结构

```
VoxBridge/
├── runtime/                    # 核心运行时（零外部依赖）
│   ├── adapter/protocol.py     # TTSAdapter 接口协议
│   ├── device/device.py        # GPU 后端检测 (Vulkan/CUDA/DirectML/CPU)
│   └── registry/registry.py    # 模型注册表
├── adapters/                   # 模型适配器实现
│   ├── qwen3/adapter.py        # Qwen3 1.7B TTS 
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

## 依赖

### Python 包

| 包 | 用途 |
|---|---|
| `numpy` | 向量计算 / 音频 / Mel |
| `onnxruntime` | ONNX 模型推理 |
| `tokenizers` (HuggingFace) | 文本 tokenize |
| `soundfile` | WAV 读写 |
| `scipy` | Mel 提取 (STFT) / 重采样 |
| `streamvox` | llama.cpp Python 绑定 (GGUF 推理) |

### 系统依赖

| 依赖 | 说明 |
|---|---|
| `ggml-vulkan.dll` | StreamVox 安装时自带，Vulkan GPU 加速 |
| `ggml-cpu.dll` | CPU 后备推理 |
| `onnxruntime.dll` | ONNX Runtime 运行时 |
| `DirectML.dll` | Windows DirectX 12 GPU 算子 |

**Windows 用户**: AMD GPU 需要安装 Vulkan Runtime (Mesa Vulkan 或驱动自带)。NVIDIA GPU 需要 CUDA Toolkit。

## 模型文件

Qwen3 1.7B TTS 需要以下模型文件，放入 `models/qwen3-tts-clone-1.7b-gguf/` 目录：

```
qwen3-tts-clone-1.7b-gguf/
├── tokenizer.json                          # HuggingFace Tokenizer
├── qwen3_tts_decoder.fp16.onnx             # ONNX 音频解码器 (~350MB)
├── ckpt/clone_1.7B/
│   ├── qwen3_tts_talker.q5_k.gguf          # GGUF talker 模型 (~2.5GB)
│   ├── qwen3_tts_predictor.q8_0.gguf       # GGUF 预测器 (~700MB)
│   ├── qwen3_tts_speaker_encoder.fp16.onnx  # 音色编码器
│   └── embeddings/                         # 嵌入权重表
│       ├── codec_embedding_0.npy  ~  codec_embedding_15.npy  (16个)
│       ├── text_embedding_projected.npy    # 文本→嵌入映射表 (~600MB)
│       ├── proj_weight.npy                 # 投影矩阵 (~32MB)
│       └── proj_bias.npy                   # 投影偏置
