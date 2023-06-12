# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import time
from typing import Any, List, Optional

import anthropic

from cbc.lm.base import LMEngine
from cbc.utils.python import singleton


class AnthropicLMEngine(LMEngine):
    def __init__(self, model: str):
        self._client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY", None))
        self._model = model

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        # Update prompt for chat (This really needs to be moved... but it's not worth it right now)
        prompt = prompt.replace("Summary:", "\nComplete the following sentence (ONLY say the completion):\n")

        samples = []
        for _ in range(n_completions):
            error = None
            for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                try:
                    resp = self._client.completion(
                        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                        model=self._model,
                        temperature=temperature or 0.7,
                        max_tokens_to_sample=256,
                    )
                    samples.append(resp["completion"])
                    break
                except Exception as e:
                    # Backoff and try again
                    time.sleep(backoff)
                    error = e
                    continue
            else:
                raise error

        return samples

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1, temperature=0.0)[0]


@singleton
class Claude(AnthropicLMEngine):
    def __init__(self):
        super().__init__("claude-1")


@singleton
class Claude100K(AnthropicLMEngine):
    def __init__(self):
        super().__init__("claude-1-100k")


@singleton
class ClaudeInstant(AnthropicLMEngine):
    def __init__(self):
        super().__init__("claude-instant-1")


@singleton
class ClaudeInstant100K(AnthropicLMEngine):
    def __init__(self):
        super().__init__("claude-instant-1-100k")
