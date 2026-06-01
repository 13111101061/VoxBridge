from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass
class VoiceProfile:
    id: str
    name: str
    model: str
    sample_rate: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PromptData:
    model_name: str
    embedding: Any = None
    codes: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateBag:
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value


@dataclass
class ModelInfo:
    name: str
    sample_rate: int
    description: str
    supports_voice_clone: bool = True
    supports_streaming: bool = True
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0


class TTSAdapter(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def load(self, *, device: str = "auto") -> None: ...

    def stream(
        self,
        text: str,
        prompt: PromptData | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]: ...

    def shutdown(self) -> None: ...