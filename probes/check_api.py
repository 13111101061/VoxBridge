"""Check return types of load_model / create_context."""
import streamvox.models.qwen3.llama as lm
import numpy as np

lm.init_llama_lib()

MODEL_PATH = r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf\ckpt\clone_1.7B\qwen3_tts_talker.q5_k.gguf"

m = lm.load_model(MODEL_PATH, n_gpu_layers=-1, backend="vulkan")
print(f"load_model: type={type(m).__name__}")
for attr in ["model", "ctx", "n_embd", "n_vocab", "_model", "_handle"]:
    if hasattr(m, attr):
        print(f"  .{attr} = {type(getattr(m, attr)).__name__}")

# Check if llama_model_n_embd accepts m
try:
    r = lm.llama_model_n_embd(m)
    print(f"llama_model_n_embd(m): {r}")
except Exception as e:
    print(f"llama_model_n_embd(m): FAILED - {e}")

# Check if we can get n_embd from the model object
if hasattr(m, 'n_embd'):
    print(f"m.n_embd: {m.n_embd}")

# Check context
ctx = lm.create_context(m, n_ctx=256, n_batch=256, embeddings=True, no_perf=True)
print(f"\ncreate_context: type={type(ctx).__name__}")
for attr in ["ctx", "model", "n_ctx"]:
    if hasattr(ctx, attr):
        print(f"  .{attr} = {type(getattr(ctx, attr)).__name__}")

# Check LlamaContext class
if hasattr(lm, 'LlamaContext'):
    print(f"\nLlamaContext class: {lm.LlamaContext}")
    try:
        sig = __import__('inspect').signature(lm.LlamaContext)
        print(f"  signature: {sig}")
    except Exception:
        pass

# Check LlamaModel class
if hasattr(lm, 'LlamaModel'):
    print(f"\nLlamaModel class: {lm.LlamaModel}")
