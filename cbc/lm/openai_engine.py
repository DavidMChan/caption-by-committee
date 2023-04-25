import os
import time
from typing import Any, List, Optional

import openai

from cbc.lm.base import LMEngine
from cbc.utils.python import singleton

# Setup openai api keys
openai.organization = os.getenv("OPENAI_API_ORG", None)
openai.api_key = os.getenv("OPENAI_API_KEY", None)


class OpenAI:
    USAGE = 0


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
                OpenAI.USAGE += int(cp.usage.total_tokens) * self.COST_PER_TOKEN
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

        # Filter the prompt (one-off experiment)
        prompt = prompt.replace("Summary:", "\nComplete the following sentence according to the task above:\nSummary: ")

        error = None
        for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                cp = openai.ChatCompletion.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant, which does tasks as instructed by the user. You never make any references to yourself, and you never use the word 'I'. You never mention the reasoning behind your answers, but you always think through it carefully, before giving a concise and accurate reply. You never mention captions, or make reference to the messages given by the user.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                    n=n_completions,
                    **kwargs,
                )  # type: ignore
                OpenAI.USAGE += int(cp.usage.total_tokens) * self.COST_PER_TOKEN
                return [i.message.content for i in cp.choices]  # type: ignore
            except Exception as e:
                # Backoff and try again
                time.sleep(backoff)
                error = e
                continue

        raise error

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]


@singleton
class ChatGPT(OpenAIChatEngine):
    COST_PER_TOKEN = 0.002 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-3.5-turbo")


@singleton
class GPT4(OpenAIChatEngine):
    COST_PER_TOKEN = 0.03 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-4")


@singleton
class GPT432K(OpenAIChatEngine):
    COST_PER_TOKEN = 0.06 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-4-32k")


@singleton
class GPT3Davinci3(OpenAILMEngine):
    COST_PER_TOKEN = 0.02 / 1000

    def __init__(self) -> None:
        super().__init__("text-davinci-003")


@singleton
class GPT3Davinci2(OpenAILMEngine):
    COST_PER_TOKEN = 0.03 / 1000

    def __init__(self) -> None:
        super().__init__("text-davinci-002")


@singleton
class GPT3Curie(OpenAILMEngine):
    COST_PER_TOKEN = 0.002 / 1000

    def __init__(self) -> None:
        super().__init__("text-curie-001")


@singleton
class GPT3Babbage(OpenAILMEngine):
    COST_PER_TOKEN = 0.02 / 1000

    def __init__(self) -> None:
        super().__init__("text-babbage-001")


@singleton
class GPT3Ada(OpenAILMEngine):
    COST_PER_TOKEN = 0.02 / 1000

    def __init__(self) -> None:
        super().__init__("text-ada-001")
