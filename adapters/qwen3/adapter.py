"""
Qwen3 1.7B TTS Adapter — 流式推理全链路
使用 StreamVox LlamaBatch + ONNX Runtime，零商业依赖。
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterator

import streamvox.models.qwen3.llama as lm

from runtime.adapter.protocol import ModelInfo, PromptData
from runtime.device.device import DeviceContext, detect_device

# ─── constants ────────────────────────────────────────────────────
SAMPLE_RATE: int = 24000; CHUNK_SIZE: int = 12; SKIP_SAMPLES: int = 7680
N_CODE_GROUPS: int = 16; MAX_NEW_TOKENS: int = 2048
IM_START: int = 151644; IM_END: int = 151645
CODE_EOS: int = 4198

# from llama_model_n_embd()
N_EMBD_T: int = 2048
N_EMBD_P: int = 1024

S = dict(temperature=0.9, top_p=1.0, top_k=50, seed=42)


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
        self._talker_m: int = 0; self._talker_ctx: int = 0
        self._pred_m: int = 0; self._pred_ctx: int = 0
        self._n_embd_t: int = N_EMBD_T; self._n_embd_p: int = N_EMBD_P
        self._decoder_sess: object = None
        self._embeddings: list[np.ndarray] = []
        self._proj_w: np.ndarray | None = None
        self._proj_b: np.ndarray | None = None
        self._rng = np.random.default_rng(42)
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return self._info

    # ── load ─────────────────────────────────────────────────

    def load(self, *, device: str = "auto") -> None:
        if self._loaded: return
        dev = detect_device(device)
        lm.init_llama_lib()
        bb = dev.llama_backend
        self._talker_m = lm.load_model(str(self._ckpt / "qwen3_tts_talker.q5_k.gguf"), n_gpu_layers=-1, backend=bb)
        self._talker_ctx = lm.create_context(self._talker_m, n_ctx=4096, n_batch=2048, embeddings=True, no_perf=True)
        self._pred_m = lm.load_model(str(self._ckpt / "qwen3_tts_predictor.q8_0.gguf"), n_gpu_layers=-1, backend=bb)
        self._pred_ctx = lm.create_context(self._pred_m, n_ctx=512, n_batch=2048, embeddings=True, no_perf=True)
        self._load_assets()
        self._load_onnx(dev)
        self._loaded = True

    def _load_assets(self) -> None:
        d = self._ckpt / "embeddings"
        self._embeddings = [np.load(str(d / f"codec_embedding_{i}.npy")) for i in range(N_CODE_GROUPS)]
        self._proj_w = np.load(str(d / "proj_weight.npy"))
        self._proj_b = np.load(str(d / "proj_bias.npy"))

    def _load_onnx(self, dev: DeviceContext) -> None:
        import onnxruntime as ort
        prov = [dev.ort_provider, "CPUExecutionProvider"]
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cr = self._ckpt.parent
        self._decoder_sess = ort.InferenceSession(str(cr / "qwen3_tts_decoder.fp16.onnx"), sess_options=so, providers=prov)

    # ── public ───────────────────────────────────────────────

    def stream(self, text: str, prompt: PromptData | None = None,
               *, language: str = "chinese", **kw) -> Iterator[np.ndarray]:
        self._ensure()
        tokens = self._tokenize(text)
        codes = self._invoke_talker(tokens)
        latent = self._project(codes)
        yield from self._decode_stream(latent)

    def shutdown(self) -> None:
        if self._talker_ctx: lm.llama_free(self._talker_ctx)
        if self._talker_m: lm.llama_model_free(self._talker_m)
        if self._pred_ctx: lm.llama_free(self._pred_ctx)
        if self._pred_m: lm.llama_model_free(self._pred_m)
        self._loaded = False

    # ── tokenizer ────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(str(self._ckpt.parent / "tokenizer.json"))
        nl = tok.encode("\n").ids
        assist = tok.encode("assistant").ids
        text_ids = tok.encode(text).ids
        return [IM_START] + assist + nl + text_ids + nl + [IM_END] + nl + [IM_START] + assist + nl

    # ── talker ───────────────────────────────────────────────

    def _invoke_talker(self, tokens: list[int]) -> np.ndarray:
        ctx = self._talker_ctx; n_embd = self._n_embd_t
        all_codes: list[list[int]] = []

        # ── prefill ──
        n = len(tokens)
        pb = lm.LlamaBatch(n_tokens=n, embd_dim=0, n_seq_max=1)
        for i, tid in enumerate(tokens):
            pb.token[i] = tid; pb.pos[i] = i; pb.n_seq_id[i] = 1
            pb.seq_id[i][0] = 0; pb.logits[i] = (i == n - 1)
        lm.llama_decode(ctx, pb.struct)
        pos = n

        # ── generate ──
        for _ in range(MAX_NEW_TOKENS):
            lp = lm.llama_get_logits_ith(ctx, pos - 1)
            logits = np.array(list(lp[:CODE_EOS]) + [lp[CODE_EOS]], dtype=np.float64)
            c0 = _sample(logits, rng=self._rng)
            if c0 == CODE_EOS: break

            # hidden state for predictor
            emb_all = lm.llama_get_embeddings(ctx)
            hidden = np.array(emb_all[-n_embd:]).copy().astype(np.float32)

            sub = self._run_predictor(hidden, c0)
            cg = [c0] + sub; all_codes.append(cg)

            # summed embedding (cls codec_0~15)
            summed = np.zeros(n_embd, dtype=np.float32)
            for i in range(N_CODE_GROUPS):
                summed += self._embeddings[i][cg[i]][:n_embd]

            eb = lm.LlamaBatch(n_tokens=1, embd_dim=n_embd, n_seq_max=1)
            eb.embd[0] = summed.astype(np.float32)
            eb.pos[0] = pos; eb.n_seq_id[0] = 1
            eb.seq_id[0][0] = 0; eb.logits[0] = True
            lm.llama_decode(ctx, eb.struct)
            pos += 1

        if not all_codes: raise RuntimeError("no codes")
        return np.array(all_codes, dtype=np.int32)

    # ── predictor ────────────────────────────────────────────

    def _run_predictor(self, hidden: np.ndarray, code_0: int) -> list[int]:
        ctx = self._pred_ctx; n_embd = self._n_embd_p
        c0_emb = self._embeddings[0][code_0].astype(np.float32)[:n_embd]
        combined = np.vstack([hidden.reshape(1, -1), c0_emb.reshape(1, -1)]).astype(np.float32)

        pb = lm.LlamaBatch(n_tokens=2, embd_dim=n_embd, n_seq_max=1)
        pb.embd[0] = combined[0]; pb.embd[1] = combined[1]
        pb.pos[0] = 0; pb.pos[1] = 1
        pb.n_seq_id[0] = 1; pb.n_seq_id[1] = 1
        pb.seq_id[0][0] = 0; pb.seq_id[1][0] = 0
        pb.logits[0] = False; pb.logits[1] = True
        lm.llama_decode(ctx, pb.struct)
        cur = 2

        sub: list[int] = []
        for idx in range(1, N_CODE_GROUPS):
            lp = lm.llama_get_logits_ith(ctx, cur - 1)
            nxt = _sample(np.array(lp[:2048], dtype=np.float64), rng=self._rng)
            sub.append(nxt)

            if idx < N_CODE_GROUPS - 1:
                emb = self._embeddings[idx][nxt].astype(np.float32)[:n_embd]
                nb = lm.LlamaBatch(n_tokens=1, embd_dim=n_embd, n_seq_max=1)
                nb.embd[0] = emb.astype(np.float32)
                nb.pos[0] = cur; nb.n_seq_id[0] = 1
                nb.seq_id[0][0] = 0; nb.logits[0] = True
                lm.llama_decode(ctx, nb.struct)
                cur += 1

        return sub

    # ── projection ───────────────────────────────────────────

    def _project(self, codes: np.ndarray) -> np.ndarray:
        T = codes.shape[0]; dim = self._embeddings[0].shape[1]
        acc = np.zeros((T, dim), dtype=np.float32)
        for i in range(N_CODE_GROUPS):
            acc += self._embeddings[i][codes[:, i]]
        return acc @ self._proj_w.T + self._proj_b

    # ── decoder ──────────────────────────────────────────────

    def _decode_stream(self, latent: np.ndarray) -> Iterator[np.ndarray]:
        sess = self._decoder_sess
        state = _init_decoder_state()
        T = latent.shape[0]
        for i in range(0, T, CHUNK_SIZE):
            chunk = latent[i:i + CHUNK_SIZE]; n = chunk.shape[0]
            if n < CHUNK_SIZE:
                chunk = np.concatenate([chunk, np.zeros((CHUNK_SIZE - n, latent.shape[1]), dtype=latent.dtype)])
            is_last = np.array([1.0 if i + CHUNK_SIZE >= T else 0.0], dtype=np.float16)
            feed = {"audio_codes": chunk[None].astype(np.float16),
                    "pre_conv_history": state["pre_conv_history"],
                    "latent_buffer": state["latent_buffer"],
                    "conv_history": state["conv_history"], "is_last": is_last}
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
            if state.get("_first"): state["_first"] = False; yield wave
            else: yield wave[SKIP_SAMPLES:]

    def _ensure(self) -> None:
        if not self._loaded: raise RuntimeError("call .load() first")


# ─── utilities ────────────────────────────────────────────────────

def _sample(logits: np.ndarray, temperature: float = 0.9, top_p: float = 1.0,
            top_k: int = 50, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng(42)
    logits = logits.astype(np.float64); logits = logits - logits.max()
    if temperature <= 0: return int(np.argmax(logits))
    logits = logits / temperature
    ex = np.exp(logits - logits.max()); probs = ex / ex.sum()
    if 0 < top_k < len(probs):
        idx = np.argpartition(-probs, top_k)[:top_k]
        m = np.zeros_like(probs); m[idx] = 1.0; probs *= m
    if top_p < 1.0:
        si = np.argsort(-probs); cu = np.cumsum(probs[si])
        ct = np.searchsorted(cu, top_p, side='right')
        m = np.zeros_like(probs); m[si[:ct + 1]] = 1.0; probs *= m
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _init_decoder_state() -> dict:
    st = {"pre_conv_history": np.zeros((1, 512, 2), dtype=np.float16),
          "latent_buffer": np.zeros((1, 1024, 4), dtype=np.float16),
          "conv_history": np.zeros((1, 1024, 4), dtype=np.float16), "_first": True}
    for i in range(8):
        st[f"past_key_{i}"] = np.zeros((1, 16, 0, 64), dtype=np.float16)
        st[f"past_value_{i}"] = np.zeros((1, 16, 0, 64), dtype=np.float16)
    return st