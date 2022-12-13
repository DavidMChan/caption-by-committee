from abc import ABC, abstractmethod
from typing import Any, List, Optional


class LMEngine(ABC):
    @abstractmethod
    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def best(self, prompt: str) -> str:
        raise NotImplementedError()
