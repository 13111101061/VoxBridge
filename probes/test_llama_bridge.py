"""Quick test: verify full ctypes llama bridge works with StreamVox DLL."""
import ctypes, os, pathlib, sys, numpy as np

sys.path.insert(0, r"E:\D G\AI\AI-TTS")

BIN = pathlib.Path(r"E:\D G\H D\StreamVox-master\.venv\Lib\site-packages\streamvox\bin")
MODEL_CKPT = pathlib.Path(r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf\ckpt\clone_1.7B")

os.add_dll_directory(str(BIN))

# Use StreamVox's init for backend loading
from streamvox.models.qwen3.llama import init_llama_lib, load_model, create_context
from streamvox.models.qwen3.llama import llama_model_n_embd, llama_model_get_vocab

init_llama_lib()
print("[OK] backends inited via StreamVox")

# Load model
talker_path = str(MODEL_CKPT / "qwen3_tts_talker.q5_k.gguf")
pred_path = str(MODEL_CKPT / "qwen3_tts_predictor.q8_0.gguf")

talker_model = load_model(talker_path, n_gpu_layers=-1, backend="vulkan")
talker_ctx = create_context(talker_model, n_ctx=512, n_batch=2048, no_perf=True)
print(f"talker: model=0x{talker_model:x}, ctx=0x{talker_ctx:x}")

pred_model = load_model(pred_path, n_gpu_layers=-1, backend="vulkan")
pred_ctx = create_context(pred_model, n_ctx=512, n_batch=2048, no_perf=True)
print(f"predictor: model=0x{pred_model:x}, ctx=0x{pred_ctx:x}")

n_embd = llama_model_n_embd(talker_model)
print(f"talker n_embd = {n_embd}")

# Now use ctypes for batch operations
dll = ctypes.CDLL(str(BIN / "llama.dll"))

# Already loaded, just bind
dll.llama_batch_init.restype = ctypes.c_void_p
dll.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
dll.llama_batch_free.restype = None
dll.llama_batch_free.argtypes = [ctypes.c_void_p]
dll.llama_decode.restype = ctypes.c_int32
dll.llama_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# Test tokenizer
vocab = llama_model_get_vocab(talker_model)
dll.llama_tokenize.restype = ctypes.c_int32
dll.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                               ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]

test_text = b"hello world test"
buf = (ctypes.c_int32 * 100)()
n = dll.llama_tokenize(vocab, test_text, len(test_text), buf, 100, False, False)
print(f"tokenize 'hello world test': {[int(buf[i]) for i in range(n)]}")

# Test embedding batch
b = dll.llama_batch_init(1, n_embd, 1)
print(f"batch_init(1,{n_embd},1) = 0x{b:x}")

# Manually write embd and run decode
emb_data = np.zeros(n_embd, dtype=np.float32)
emb_data[0] = 1.0  # non-zero to see something
embd_ptr = emb_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
ctypes.memmove(b, embd_ptr, n_embd * 4)  # this writes to wrong offset!

# Actually need to access correct offset. Let's check batch struct layout.
# Modern llama_batch layout:
#   int32_t n_tokens
#   llama_token * token  
#   float * embd
#   
# On 64-bit: n_tokens=4bytes, padding=4bytes, token=8bytes, embd=8bytes
# So embd is at offset 16

print("\nTesting batch struct via offset...")
# n_tokens at offset 0
ctypes.cast(b, ctypes.POINTER(ctypes.c_int32))[0] = 1
# embd at offset 16 (int32 + 4pad + ptr = 16 on 64-bit)
# Actually let me just verify with a known working batch

# Let me instead try the token-only path first
btok = dll.llama_batch_init(1, 0, 1)
print(f"token batch = 0x{btok:x}")

# Write token
p_tok = ctypes.cast(btok, ctypes.POINTER(ctypes.c_int32))
p_tok[0] = 1  # n_tokens

# token ptr at offset 8 (after n_tokens + padding)
tok_ptr = ctypes.cast(ctypes.c_void_p(btok + 8), ctypes.POINTER(ctypes.c_void_p))
# Actually this is getting complex without the struct definition

print("\nNeed proper struct definition for batch access")
print("switching to StreamVox's built-in batch functions...")
dll.llama_batch_free(b)
dll.llama_batch_free(btok)

# StreamVox has llama_batch as a PyCStructType - let's use that
from streamvox.models.qwen3.llama import llama_batch, llama_pos, llama_seq_id, llama_token as ll_token
# These are ctypes structure types
batch_cls = llama_batch
print(f"llama_batch fields: {[(f[0], f[1]) for f in batch_cls._fields_] if hasattr(batch_cls, '_fields_') else 'no fields'}")

# Test with StreamVox's batch struct
if hasattr(batch_cls, '_fields_'):
    fields = batch_cls._fields_
    print(f"batch struct fields: {[f[0] for f in fields]}")

print("\n[DONE]")
