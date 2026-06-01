"""Test raw llama_batch_init with embedding injection via StreamVox C API."""
import numpy as np
import ctypes
import sys

sys.path.insert(0, r"E:\D G\H D\StreamVox-master\.venv\Lib\site-packages")
import streamvox.models.qwen3.llama as lm

lm.init_llama_lib()

MODEL_PATH = r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf\ckpt\clone_1.7B\qwen3_tts_talker.q5_k.gguf"
EMBED_PATH = r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf\ckpt\clone_1.7B\embeddings\text_embedding_projected.npy"

m = lm.LlamaModel(MODEL_PATH, n_gpu_layers=0)
ctx = lm.LlamaContext(m, n_ctx=512, n_batch=512, embeddings=True, no_perf=True)
n_embd = m.n_embd
print(f"n_embd={n_embd}, n_vocab={m.n_vocab}")

# ---------- raw llama_batch_init (C API, no Python wrapper) ----------
batch = lm.llama_batch_init(3, n_embd, 1)
print(f"batch.n_tokens (before set)={batch.n_tokens}")
print(f"batch.embd allocated={bool(batch.embd)}")
print(f"n_tokens ctype={type(batch.n_tokens).__name__}")

# C API's llama_batch_init only ALLOCATES arrays; n_tokens starts at 0
batch.n_tokens = 1

# Fill embedding for first position
te = np.load(EMBED_PATH)
vec = te[14990].astype(np.float32)  # "hello"
ctypes.memmove(batch.embd, vec.ctypes.data, n_embd * 4)
print(f"embedding filled: [{vec[0]:.4f}, {vec[1]:.4f}, ..., {vec[-1]:.4f}]")

batch.pos[0] = 0
batch.logits[0] = True
batch.n_seq_id[0] = 1
batch.seq_id[0][0] = 0

print(f"\n--- Calling ctx.decode ---")
rc = ctx.decode(batch)
print(f"decode rc = {rc}")

if rc == 0:
    logits = ctx.get_logits_ith(0)
    print(f"\nlogits: shape={logits.shape}, dtype={logits.dtype}")
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"top5 token ids: {top5}")
    print(f"top5 values:   {[logits[i] for i in top5]}")
    print(f"codec_EOS(4198) = {logits[4198]:.4f}")
    print(f"codec_BOS(4199) = {logits[4199]:.4f}")
    
    # embeddings output (hidden states)
    emb = ctx.get_embeddings()
    print(f"\nembeddings: shape={emb.shape}, nonzero in last pos={np.count_nonzero(emb[-1])}")
else:
    print("\n--- DECODE FAILED ---")

lm.llama_batch_free(batch)
