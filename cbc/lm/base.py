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

    @staticmethod
    def from_string(typestr: str, **kwargs: Any) -> "LMEngine":
        from cbc.lm import LM_ENGINES, LM_ENGINES_CLI

        if typestr in LM_ENGINES:
            return LM_ENGINES[typestr](**kwargs)  # type: ignore
        elif typestr in LM_ENGINES_CLI:
            return LM_ENGINES_CLI[typestr](**kwargs)  # type: ignore
        raise ValueError(f"Invalid language model type: {typestr}")
