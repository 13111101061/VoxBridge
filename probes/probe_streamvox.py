"""
StreamVox 中间张量探针脚本
目标: dump 推理管线每一步的 tensor shape / dtype / 数值范围
输出: probes/output/ 目录下 .npz + probe_report.json
"""
from __future__ import annotations

import json
import numpy as np
import os
import soundfile as sf
from pathlib import Path

from streamvox import TTSEngine

OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_DIR = Path(r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf")
REF_AUDIO = Path(r"E:\D G\H D\StreamVox-master\example\Condition3.wav")
REF_TEXT = "所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。"
TEST_TEXT = "你好，这是探针测试。"
OUTPUT_DIR.mkdir(exist_ok=True)


def probe_onnx_models():
    """Step 0: 探查所有 ONNX 模型的输入/输出签名"""
    import onnxruntime as ort

    model_dir = MODEL_DIR / "ckpt"

    onnx_files = {
        "speaker_encoder": model_dir / "clone_1.7B" / "qwen3_tts_speaker_encoder.fp16.onnx",
        "codec_encoder": model_dir / "qwen3_tts_codec_encoder.fp16.onnx",
        "decoder": model_dir / "qwen3_tts_decoder.fp16.onnx",
    }

    report = {}
    for name, path in onnx_files.items():
        if not path.exists():
            print(f"[SKIP] {name}: file not found at {path}")
            continue

        print(f"\n=== ONNX Model: {name} ===")
        print(f"  Path: {path}")

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        inputs = {}
        for inp in sess.get_inputs():
            inputs[inp.name] = {
                "shape": inp.shape,
                "type": str(inp.type),
            }
            print(f"  INPUT:  name={inp.name}, shape={inp.shape}, type={inp.type}")

        outputs = {}
        for out in sess.get_outputs():
            outputs[out.name] = {
                "shape": out.shape,
                "type": str(out.type),
            }
            print(f"  OUTPUT: name={out.name}, shape={out.shape}, type={out.type}")

        report[name] = {"path": str(path), "inputs": inputs, "outputs": outputs}

    return report


def probe_embeddings():
    """Step 1: 探查 16 层 codec embedding + projection 的 shape"""
    model_dir = MODEL_DIR / "ckpt" / "clone_1.7B" / "embeddings"

    report = {}
    files = sorted(model_dir.glob("*.npy"))

    print("\n=== Embedding Files ===")
    for f in files:
        arr = np.load(str(f))
        key = f.stem
        report[key] = {
            "path": str(f),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.4f}, {arr.max():.4f}]")
        np.savez(str(OUTPUT_DIR / f"emb_{key}.npz"), data=arr)

    return report


def probe_streamvox_runtime():
    """Step 2: 用 StreamVox 跑一次推理，dump 中间张量"""
    engine = TTSEngine(
        model=str(MODEL_DIR),
        device="auto",
        verify_model_sha256=False,
    )

    prompt = engine.make_prompt(
        role_name="probe_role",
        audio_path=str(REF_AUDIO),
        prompt_text=REF_TEXT,
        persist=False,
    )

    runtime = engine.runtime

    report = {}
    report["sample_rate"] = runtime.sample_rate
    report["runtime_info"] = {
        "model": runtime.runtime_info.model,
        "adapter": runtime.runtime_info.adapter,
        "bundle_loaded": runtime.runtime_info.bundle_loaded,
    }

    print(f"\n=== Runtime Info ===")
    print(f"  sample_rate: {runtime.sample_rate}")
    print(f"  model: {runtime.runtime_info.model}")
    print(f"  adapter: {runtime.runtime_info.adapter}")

    # Probe prompt internal data
    prompt_asset = prompt
    if hasattr(prompt_asset, 'speaker_embedding'):
        spk_emb = prompt_asset.speaker_embedding
        report["speaker_embedding"] = {
            "shape": list(spk_emb.shape),
            "dtype": str(spk_emb.dtype),
            "range": [float(spk_emb.min()), float(spk_emb.max())],
        }
        print(f"\n  speaker_embedding: shape={spk_emb.shape}, dtype={spk_emb.dtype}")
        np.savez(str(OUTPUT_DIR / "speaker_embedding.npz"), data=spk_emb)

    if hasattr(prompt_asset, 'prompt_codes'):
        codes = prompt_asset.prompt_codes
        report["prompt_codes"] = {
            "shape": list(codes.shape),
            "dtype": str(codes.dtype),
            "range": [float(codes.min()), float(codes.max())],
            "unique_values": len(np.unique(codes)),
        }
        print(f"  prompt_codes: shape={codes.shape}, dtype={codes.dtype}, unique={len(np.unique(codes))}")
        np.savez(str(OUTPUT_DIR / "prompt_codes.npz"), data=codes)

    if hasattr(prompt_asset, 'decoder_state') and prompt_asset.decoder_state is not None:
        state = prompt_asset.decoder_state
        state_report = {}
        for attr in ['kv_cache', 'conv_history', 'pre_conv_history', 'latent_buffer', 'skip_samples']:
            if hasattr(state, attr):
                val = getattr(state, attr)
                if isinstance(val, np.ndarray):
                    state_report[attr] = {
                        "shape": list(val.shape),
                        "dtype": str(val.dtype),
                    }
                    np.savez(str(OUTPUT_DIR / f"prompt_state_{attr}.npz"), data=val)
                elif isinstance(val, list):
                    state_report[attr] = {
                        "type": "list",
                        "length": len(val),
                        "item_shapes": [list(v.shape) if isinstance(v, np.ndarray) else str(type(v)) for v in val],
                        "item_dtypes": [str(v.dtype) if isinstance(v, np.ndarray) else str(type(v)) for v in val],
                    }
                    for i, v in enumerate(val):
                        if isinstance(v, np.ndarray):
                            np.savez(str(OUTPUT_DIR / f"prompt_state_{attr}_{i}.npz"), data=v)
                elif isinstance(val, int):
                    state_report[attr] = {"type": "int", "value": val}
                else:
                    state_report[attr] = {"type": str(type(val)), "value": str(val)}
        report["prompt_decoder_state"] = state_report
        print(f"  decoder_state: {state_report}")

    # Run stream and capture audio chunks + their shapes
    print("\n=== Stream Output ===")
    chunks_data = []
    chunk_idx = 0
    for chunk in engine.stream(
        text=TEST_TEXT,
        role_name=prompt,
        language="chinese",
        track_performance=True,
    ):
        chunks_data.append(chunk)
        print(f"  chunk[{chunk_idx}]: shape={chunk.shape}, dtype={chunk.dtype}, range=[{chunk.min():.4f}, {chunk.max():.4f}]")
        np.savez(str(OUTPUT_DIR / f"audio_chunk_{chunk_idx}.npz"), data=chunk)
        chunk_idx += 1

    audio = np.concatenate(chunks_data, axis=-1)
    report["audio_output"] = {
        "total_samples": len(audio),
        "total_duration_s": len(audio) / runtime.sample_rate,
        "num_chunks": chunk_idx,
        "chunk_shapes": [list(c.shape) for c in chunks_data],
        "dtype": str(audio.dtype),
    }

    sf.write(str(OUTPUT_DIR / "probe_output.wav"), audio, runtime.sample_rate)
    print(f"  total: {len(audio)} samples = {len(audio)/runtime.sample_rate:.2f}s audio")

    engine.shutdown()
    return report


def probe_tokenizer():
    """Step 3: 探查 tokenizer 和 special tokens"""
    from streamvox.models.qwen3.schema.constant import SPECIAL_TOKENS, LANGUAGE_MAP

    report = {
        "special_tokens": {k: int(v) for k, v in SPECIAL_TOKENS.items()},
        "language_map": {k: int(v) for k, v in LANGUAGE_MAP.items()},
    }

    print("\n=== Special Tokens ===")
    for k, v in SPECIAL_TOKENS.items():
        print(f"  {k}: {v}")

    print("\n=== Language Map ===")
    for k, v in LANGUAGE_MAP.items():
        print(f"  {k}: {v}")

    return report


def main():
    print("=" * 60)
    print("StreamVox Tensor Probe - collecting intermediate data")
    print("=" * 60)

    full_report = {}

    print("\n[1/4] Probing ONNX model signatures...")
    full_report["onnx_models"] = probe_onnx_models()

    print("\n[2/4] Probing embedding files...")
    full_report["embeddings"] = probe_embeddings()

    print("\n[3/4] Probing tokenizer & special tokens...")
    full_report["tokenizer"] = probe_tokenizer()

    print("\n[4/4] Probing StreamVox runtime (full inference)...")
    full_report["runtime"] = probe_streamvox_runtime()

    report_path = OUTPUT_DIR / "probe_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {report_path}")
    print(f"Tensors saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()