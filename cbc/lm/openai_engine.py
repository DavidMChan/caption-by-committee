import os
from typing import Any, List, Optional

import openai
import time

from cbc.lm.base import LMEngine

# Setup openai api keys
openai.organization = os.getenv("OPENAI_API_ORG", None)
openai.api_key = os.getenv("OPENAI_API_KEY", None)


class OpenAILMEngine(LMEngine):
    def __init__(self, model: str):
        self._model = model

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        error = None
        for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                cp = openai.Completion.create(
                    model=self._model, prompt=prompt, temperature=temperature, max_tokens=256, n=n_completions, **kwargs
                )  # type: ignore
                return [i.text for i in cp.choices]  # type: ignore
            except Exception as e:
                # Backoff and try again
                time.sleep(backoff)
                error = e
                continue

        raise error

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]

class OpenAIChatEngine(LMEngine):
    def __init__(self, model: str):
        self._model = model

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        error = None
        for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                cp = openai.ChatCompletion.create(
                    model=self._model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant which can complete sentences for the user.'},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=temperature,
                    max_tokens=256,
                    n=n_completions,
                    **kwargs
                )  # type: ignore
                return [i.message.content for i in cp.choices]  # type: ignore
            except Exception as e:
                # Backoff and try again
                time.sleep(backoff)
                error = e
                continue

        raise error

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]


class ChatGPT(OpenAIChatEngine):
    def __init__(self) -> None:
        super().__init__("gpt-3.5-turbo")

class GPT3Davinci3(OpenAILMEngine):
    def __init__(self) -> None:
        super().__init__("text-davinci-003")


class GPT3Davinci2(OpenAILMEngine):
    def __init__(self) -> None:
        super().__init__("text-davinci-002")


class GPT3Curie(OpenAILMEngine):
    def __init__(self) -> None:
        super().__init__("text-curie-001")


class GPT3Babbage(OpenAILMEngine):
    def __init__(self) -> None:
        super().__init__("text-babbage-001")


class GPT3Ada(OpenAILMEngine):
    def __init__(self) -> None:
        super().__init__("text-ada-001")
