import os
import time
from typing import Any, List, Optional

import requests

from cbc.lm.base import LMEngine


class InferenceError(RuntimeError):
    pass


class HuggingfaceInferenceLMEngine(LMEngine):
    def __init__(self, model: str):
        self._model = model

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        if temperature is not None:
            raise InferenceError("Huggingface API does not support temperature")

        completions = []
        for _ in range(n_completions):
            for _ in range(10):
                API_URL = f"https://api-inference.huggingface.co/models/{self._model}"
                headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "do_sample": False if n_completions == 1 else True,
                            "wait_for_model": True,
                        },
                    },
                ).json()

                if "error" in response:
                    if "loading" in response["error"]:
                        print("Model is still loading, retrying in 60s...")
                        time.sleep(60)
                        continue
                    raise InferenceError(response["error"])

                outputs = [
                    g["generated_text"].replace(prompt, "").split(".")[0].strip().replace("\n", " ") + "."
                    for g in response
                ]
                completions.append(outputs[0])
                break
            else:
                raise InferenceError("Failed to generate completions: Model is still loading")

        return completions

    def best(self, prompt: str) -> str:
        return self(prompt, n_completions=1)[0]


class Bloom(HuggingfaceInferenceLMEngine):
    def __init__(self) -> None:
        super().__init__("bigscience/bloom")


class OPT(HuggingfaceInferenceLMEngine):
    def __init__(self) -> None:
        super().__init__("facebook/opt-66b")
