"""
Qwen3 1.7B TTS Adapter — 流式推理全链路
使用 StreamVox llama.cpp DLL + HuggingFace tokenizer + ONNX Runtime，零商业依赖。
"""
from __future__ import annotations

import ctypes
import numpy as np
from pathlib import Path
from typing import Iterator

import streamvox.models.qwen3.llama as lm

from runtime.adapter.protocol import ModelInfo, PromptData
from runtime.device.device import DeviceContext, detect_device

SAMPLE_RATE: int = 24000; CHUNK_SIZE: int = 12; SKIP_SAMPLES: int = 7680
N_CODE_GROUPS: int = 16; MAX_NEW_TOKENS: int = 2048
IM_START: int = 151644; IM_END: int = 151645
CODE_EOS: int = 4198

N_EMBD_T: int = 2048
N_EMBD_P: int = 2048  # predictor handles internal 1024 projection


# ─── EmbeddingBatch ────────────────────────────────────────────────

def _get_batch_type() -> type:
    """Get the llama_batch ctypes type used by StreamVox's DLL."""
    dummy_tokens = (ctypes.c_int32 * 1)(198)
    batch = lm.llama_batch_get_one(dummy_tokens, 1, 0, 0)
    return type(batch)


def _make_embd_batch(embeddings: np.ndarray, *, pos_offset: int = 0,
                     logits_mask: list[bool] | None = None) -> tuple[ctypes.Structure, list]:
    """
    Build a llama_batch with token=NULL and embd=provided data.

    Returns (batch_struct, keepalive_list) where keepalive prevents GC.
    embeddings shape: [N, n_embd] (float32)
    """
    N = embeddings.shape[0]
    assert embeddings.ndim == 2, f"expected [N, n_embd], got {embeddings.shape}"
    _type = _get_batch_type()
    keep: list = []
    batch = _type()

    # embd — use the numpy array's own memory
    emb_ptr = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    keep.append(embeddings)

    # token — NULL
    null_tok = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int32))

    # pos
    pos_arr = (ctypes.c_int32 * N)(*(pos_offset + i for i in range(N)))
    keep.append(pos_arr)

    # n_seq_id
    nsi_arr = (ctypes.c_int32 * N)(*([1] * N))
    keep.append(nsi_arr)

    # seq_id — array of pointers
    seq_ptrs = (ctypes.POINTER(ctypes.c_int32) * N)()
    for i in range(N):
        arr = (ctypes.c_int32 * 1)(0)
        seq_ptrs[i] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_int32))
        keep.append(arr)
    keep.append(seq_ptrs)

    # logits
    if logits_mask is None:
        logits_mask = [False] * (N - 1) + [True]
    lg_arr = (ctypes.c_int8 * N)(*[1 if b else 0 for b in logits_mask])
    keep.append(lg_arr)

    batch.n_tokens = N
    batch.token = null_tok
    batch.embd = emb_ptr
    batch.pos = pos_arr
    batch.n_seq_id = nsi_arr
    batch.seq_id = seq_ptrs
    batch.logits = lg_arr

    return batch, keep


# ─── adapter ──────────────────────────────────────────────────────

class Qwen3TTSAdapter:
    def __init__(self, model_dir: str) -> None:
        d = Path(model_dir)
        self._ckpt = d / "ckpt" / "clone_1.7B"
        self._info = ModelInfo(
            name="qwen3-tts-clone-1.7b-gguf", sample_rate=SAMPLE_RATE,
            description="Qwen3 TTS 1.7B Vulkan + DirectML",
            supports_voice_clone=True, supports_streaming=True,
            min_vram_gb=3.0, min_ram_gb=4.0,
        )
        self._talker_m: object = None
        self._talker_ctx: object = None
        self._pred_m: object = None
        self._pred_ctx: object = None
        self._decoder_sess: object = None
        self._embeddings: list[np.ndarray] = []
        self._text_embd: np.ndarray | None = None
        self._proj_w: np.ndarray | None = None
        self._proj_b: np.ndarray | None = None
        self._rng = np.random.default_rng(42)
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return self._info

    # ── load ─────────────────────────────────────────────────

    def load(self, *, device: str = "auto") -> None:
        if self._loaded:
            return
        dev = detect_device(device)
        lm.init_llama_lib()
        bb = dev.llama_backend
        self._talker_m = lm.load_model(
            str(self._ckpt / "qwen3_tts_talker.q5_k.gguf"),
            n_gpu_layers=-1, backend=bb,
        )
        self._talker_ctx = lm.create_context(
            self._talker_m, n_ctx=4096, n_batch=2048,
            embeddings=True, no_perf=True,
        )
        self._pred_m = lm.load_model(
            str(self._ckpt / "qwen3_tts_predictor.q8_0.gguf"),
            n_gpu_layers=-1, backend=bb,
        )
        self._pred_ctx = lm.create_context(
            self._pred_m, n_ctx=512, n_batch=2048,
            embeddings=True, no_perf=True,
        )
        self._load_assets()
        self._load_onnx(dev)
        self._loaded = True

    def _load_assets(self) -> None:
        d = self._ckpt / "embeddings"
        self._embeddings = [
            np.load(str(d / f"codec_embedding_{i}.npy"))
            for i in range(N_CODE_GROUPS)
        ]
        self._text_embd = np.load(str(d / "text_embedding_projected.npy"))
        self._proj_w = np.load(str(d / "proj_weight.npy"))
        self._proj_b = np.load(str(d / "proj_bias.npy"))

    def _load_onnx(self, dev: DeviceContext) -> None:
        import onnxruntime as ort
        prov = [dev.ort_provider, "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._decoder_sess = ort.InferenceSession(
            str(self._ckpt.parent / "qwen3_tts_decoder.fp16.onnx"),
            sess_options=so, providers=prov,
        )

    # ── public ───────────────────────────────────────────────

    def stream(self, text: str, prompt: PromptData | None = None,
               *, language: str = "chinese", **kw) -> Iterator[np.ndarray]:
        self._ensure()
        tokens = self._tokenize(text)
        codes = self._invoke_talker(tokens)
        latent = self._project(codes)
        yield from self._decode_stream(latent)

    def shutdown(self) -> None:
        if self._talker_ctx:
            lm.llama_free(self._talker_ctx)
        if self._talker_m:
            lm.llama_model_free(self._talker_m)
        if self._pred_ctx:
            lm.llama_free(self._pred_ctx)
        if self._pred_m:
            lm.llama_model_free(self._pred_m)
        self._loaded = False

    # ── tokenizer ────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(
            str(self._ckpt.parent / "tokenizer.json")
        )
        nl = tok.encode("\n").ids
        assist = tok.encode("assistant").ids
        text_ids = tok.encode(text).ids
        return (
            [IM_START] + assist + nl
            + text_ids + nl
            + [IM_END] + nl
            + [IM_START] + assist + nl
        )

    # ── talker ───────────────────────────────────────────────

    def _invoke_talker(self, tokens: list[int]) -> np.ndarray:
        ctx = self._talker_ctx; n_embd = N_EMBD_T
        all_codes: list[list[int]] = []

        # prefill: text tokens → embeddings → embd batch
        n = len(tokens)
        embs = np.array(
            [self._text_embd[tid].astype(np.float32) for tid in tokens],
            dtype=np.float32,
        )
        batch, keep = _make_embd_batch(embs)
        lm.llama_decode(ctx, batch)
        del keep, batch
        pos = n

        # autoregressive generate
        for _ in range(MAX_NEW_TOKENS):
            lp = lm.llama_get_logits_ith(ctx, pos - 1)
            logits = np.array(
                list(lp[:CODE_EOS]) + [lp[CODE_EOS]], dtype=np.float64,
            )
            c0 = _sample(logits, rng=self._rng)
            if c0 == CODE_EOS:
                break

            emb_ptr = lm.llama_get_embeddings(ctx)
            emb_all = np.ctypeslib.as_array(emb_ptr, shape=(pos * n_embd,))
            hidden = emb_all[-n_embd:].copy().astype(np.float32)

            sub = self._run_predictor(hidden, c0)
            cg = [c0] + sub
            all_codes.append(cg)

            summed = np.zeros(n_embd, dtype=np.float32)
            for i in range(N_CODE_GROUPS):
                summed += self._embeddings[i][cg[i]].astype(np.float32)

            gen_batch, gen_keep = _make_embd_batch(
                summed.reshape(1, -1), pos_offset=pos,
            )
            lm.llama_decode(ctx, gen_batch)
            del gen_keep, gen_batch
            pos += 1

        if not all_codes:
            raise RuntimeError("no codec codes generated")
        return np.array(all_codes, dtype=np.int32)

    # ── predictor ────────────────────────────────────────────

    def _run_predictor(self, hidden: np.ndarray, code_0: int) -> list[int]:
        ctx = self._pred_ctx
        c0_emb = self._embeddings[0][code_0].astype(np.float32)
        combined = np.vstack([
            hidden.reshape(1, -1),
            c0_emb.reshape(1, -1),
        ]).astype(np.float32)

        pb, pk = _make_embd_batch(combined)
        lm.llama_decode(ctx, pb)
        del pk, pb
        cur = 2

        sub: list[int] = []
        for idx in range(1, N_CODE_GROUPS):
            lp = lm.llama_get_logits_ith(ctx, cur - 1)
            nxt = _sample(np.array(lp[:2048], dtype=np.float64), rng=self._rng)
            sub.append(nxt)

            if idx < N_CODE_GROUPS - 1:
                emb = self._embeddings[idx][nxt].astype(np.float32)
                nb, nk = _make_embd_batch(emb.reshape(1, -1), pos_offset=cur)
                lm.llama_decode(ctx, nb)
                del nk, nb
                cur += 1

        return sub

    # ── projection ───────────────────────────────────────────

    def _project(self, codes: np.ndarray) -> np.ndarray:
        T, dim = codes.shape[0], self._embeddings[0].shape[1]
        acc = np.zeros((T, dim), dtype=np.float32)
        for i in range(N_CODE_GROUPS):
            acc += self._embeddings[i][codes[:, i]].astype(np.float32)
        return acc @ self._proj_w.T.astype(np.float32) + self._proj_b.astype(np.float32)

    # ── decoder ──────────────────────────────────────────────

    def _decode_stream(self, latent: np.ndarray) -> Iterator[np.ndarray]:
        sess = self._decoder_sess
        state = _init_decoder_state()
        T = latent.shape[0]
        for i in range(0, T, CHUNK_SIZE):
            chunk = latent[i:i + CHUNK_SIZE]
            n = chunk.shape[0]
            if n < CHUNK_SIZE:
                pad = np.zeros((CHUNK_SIZE - n, latent.shape[1]), dtype=latent.dtype)
                chunk = np.concatenate([chunk, pad])
            is_last = np.array(
                [1.0 if i + CHUNK_SIZE >= T else 0.0], dtype=np.float16,
            )
            feed = {
                "audio_codes": chunk[None].astype(np.float16),
                "pre_conv_history": state["pre_conv_history"],
                "latent_buffer": state["latent_buffer"],
                "conv_history": state["conv_history"],
                "is_last": is_last,
            }
            for k in range(8):
                feed[f"past_key_{k}"] = state[f"past_key_{k}"]
                feed[f"past_value_{k}"] = state[f"past_value_{k}"]
            out = sess.run(None, feed)
            names = [o.name for o in sess.get_outputs()]
            raw = out[names.index("final_wav")].flatten().astype(np.float32)
            valid = int(out[names.index("valid_samples")])
            state["pre_conv_history"] = out[names.index("next_pre_conv_history")]
            state["latent_buffer"] = out[names.index("next_latent_buffer")]
            state["conv_history"] = out[names.index("next_conv_history")]
            for k in range(8):
                state[f"past_key_{k}"] = out[names.index(f"next_key_{k}")]
                state[f"past_value_{k}"] = out[names.index(f"next_value_{k}")]
            wave = raw[:valid]
            if state.get("_first"):
                state["_first"] = False
                yield wave
            else:
                yield wave[SKIP_SAMPLES:]

    def _ensure(self) -> None:
        if not self._loaded:
            raise RuntimeError("call .load() first")


# ─── utilities ────────────────────────────────────────────────────

def _sample(logits: np.ndarray, temperature: float = 0.9, top_p: float = 1.0,
            top_k: int = 50, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng(42)
    logits = logits.astype(np.float64)
    logits = logits - logits.max()
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    ex = np.exp(logits - logits.max())
    probs = ex / ex.sum()
    if 0 < top_k < len(probs):
        idx = np.argpartition(-probs, top_k)[:top_k]
        m = np.zeros_like(probs); m[idx] = 1.0; probs *= m
    if top_p < 1.0:
        si = np.argsort(-probs); cu = np.cumsum(probs[si])
        ct = np.searchsorted(cu, top_p, side="right")
        m = np.zeros_like(probs); m[si[:ct + 1]] = 1.0; probs *= m
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _init_decoder_state() -> dict:
    st = {
        "pre_conv_history": np.zeros((1, 512, 2), dtype=np.float16),
        "latent_buffer": np.zeros((1, 1024, 4), dtype=np.float16),
        "conv_history": np.zeros((1, 1024, 4), dtype=np.float16),
        "_first": True,
    }
    for i in range(8):
        st[f"past_key_{i}"] = np.zeros((1, 16, 0, 64), dtype=np.float16)
        st[f"past_value_{i}"] = np.zeros((1, 16, 0, 64), dtype=np.float16)
    return st
