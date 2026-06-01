from __future__ import annotations

from dataclasses import dataclass
import platform


@dataclass
class DeviceContext:
    llama_backend: str
    ort_provider: str
    gpu_index: int | None


def detect_device(device: str = "auto") -> DeviceContext:
    if device == "cpu":
        return DeviceContext(
            llama_backend="cpu",
            ort_provider="CPUExecutionProvider",
            gpu_index=None,
        )

    os_name = platform.system().lower()
    has_cuda = _check_cuda()

    if device.startswith("gpu"):
        idx = _parse_gpu_index(device)
        if os_name.startswith("win") and not has_cuda:
            return DeviceContext(
                llama_backend="vulkan",
                ort_provider="DmlExecutionProvider",
                gpu_index=idx,
            )
        if has_cuda:
            return DeviceContext(
                llama_backend="cuda",
                ort_provider="CUDAExecutionProvider",
                gpu_index=idx,
            )
        return DeviceContext(
            llama_backend="vulkan",
            ort_provider="CPUExecutionProvider",
            gpu_index=idx,
        )

    if device == "auto":
        if os_name.startswith("win") and not has_cuda:
            return DeviceContext(
                llama_backend="vulkan",
                ort_provider="DmlExecutionProvider",
                gpu_index=0,
            )
        if has_cuda:
            return DeviceContext(
                llama_backend="cuda",
                ort_provider="CUDAExecutionProvider",
                gpu_index=0,
            )
        if os_name == "linux":
            return DeviceContext(
                llama_backend="vulkan",
                ort_provider="CPUExecutionProvider",
                gpu_index=0,
            )
        return DeviceContext(
            llama_backend="cpu",
            ort_provider="CPUExecutionProvider",
            gpu_index=None,
        )

    return DeviceContext(
        llama_backend="cpu",
        ort_provider="CPUExecutionProvider",
        gpu_index=None,
    )


def _check_cuda() -> bool:
    import shutil
    if shutil.which("nvidia-smi") is None:
        return False
    import os
    for marker in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        val = os.environ.get(marker)
        if val and val.lower() not in ("", "none", "void"):
            return True
    return True


def _parse_gpu_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":")[1])
        except ValueError:
            return 0
    return 0