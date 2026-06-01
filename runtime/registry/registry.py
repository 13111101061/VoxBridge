from __future__ import annotations

from typing import Any

from runtime.adapter.protocol import TTSAdapter, ModelInfo


class ModelRegistry:
    _adapters: dict[str, TTSAdapter]

    def __init__(self) -> None:
        self._adapters: dict[str, TTSAdapter] = {}

    def register(self, adapter: TTSAdapter) -> None:
        name = adapter.info.name
        if name in self._adapters:
            raise KeyError(f"model already registered: {name}")
        self._adapters[name] = adapter

    def get(self, name: str) -> TTSAdapter:
        if name not in self._adapters:
            raise KeyError(f"unknown model: {name}")
        return self._adapters[name]

    def list_models(self) -> list[ModelInfo]:
        return [a.info for a in self._adapters.values()]

    def has(self, name: str) -> bool:
        return name in self._adapters