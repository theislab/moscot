from __future__ import annotations

from typing import Any, Callable


class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class obs:
        @cprop
        def batch_key(cls) -> str:
            return "batch_key"
