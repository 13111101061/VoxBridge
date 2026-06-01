"""End-to-end test: Qwen3TTSAdapter full pipeline → .wav"""
import sys, numpy as np, soundfile as sf

sys.path.insert(0, r"E:\D G\AI\AI-TTS")

from adapters.qwen3.adapter import Qwen3TTSAdapter

MODEL_DIR = r"E:\D G\H D\StreamVox-master\models\qwen3-tts-clone-1.7b-gguf"

print("Loading adapter...")
adapter = Qwen3TTSAdapter(MODEL_DIR)
adapter.load(device="auto")

print(f"Model: {adapter.info.name}")

# Test tokenization
tokens = adapter._tokenize("hello test")
print(f"Tokenized 'hello test': {len(tokens)} tokens")
print(f"  First 10: {tokens[:10]}")
print(f"  Last 10: {tokens[-10:]}")

# Run full generation
print("\nRunning talker + predictor + decoder...")
chunks = list(adapter.stream("hello test", language="auto"))

if chunks:
    audio = np.concatenate(chunks, axis=-1)
    sf.write(r"E:\D G\AI\AI-TTS\probes\output\test_output.wav", audio, 24000)
    print(f"Generated {len(audio)} samples = {len(audio)/24000:.2f}s audio")
    print(f"Saved to probes/output/test_output.wav")
else:
    print("No audio chunks generated")

adapter.shutdown()
print("\nDone")
