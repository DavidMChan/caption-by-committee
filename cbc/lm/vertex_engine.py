# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

try:
    from vertexai.preview.language_models import TextGenerationModel
except ImportError:
    TextGenerationModel = None
from .base import LMEngine
from typing import List, Optional, Any
from ratelimit import limits, sleep_and_retry


class VertexLMEngine(LMEngine):
    def __init__(self, model: str) -> None:
        if TextGenerationModel is None:
            raise ImportError("Please install the Vertex AI SDK to use this LM engine.")

        self._model = TextGenerationModel.from_pretrained(model)
        self._parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

    @sleep_and_retry
    @limits(
        calls=40, period=60
    )  # This is the default rate limit for Vertex AI (actual rate limit is 60 calls per minute, but we'll be conservative)
    def _rate_limited_model_predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        return [
            self._rate_limited_model_predict(
                prompt, temperature=temperature if temperature is not None else 1.0, **self._parameters, **kwargs
            ).text  # type: ignore
            for _ in range(n_completions)
        ]

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]


class PaLMEngine(VertexLMEngine):
    def __init__(self) -> None:
        super().__init__("text-bison@001")
