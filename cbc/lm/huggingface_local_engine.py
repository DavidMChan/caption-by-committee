from typing import Any, List, Optional

from transformers.pipelines import pipeline

from cbc.lm.base import LMEngine


class HuggingFaceLocalLMEngine(LMEngine):
    def __init__(self, model: str, device: Optional[str] = None):
        self._generator = pipeline("text-generation", model=model, framework="pt", device=1)

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        if temperature is not None and temperature > 0:
            outputs = self._generator(
                prompt, max_length=256, num_return_sequences=n_completions, do_sample=True, temperature=temperature
            )
        elif n_completions > 1:
            outputs = self._generator(prompt, max_length=256, num_return_sequences=n_completions, do_sample=True)
        else:
            outputs = self._generator(prompt, max_length=256, num_return_sequences=n_completions, do_sample=False)

        outputs = [g["generated_text"][len(prompt) :].strip() for g in outputs]  # type: ignore

        # Hack to just use the first sentence instead of the full output
        outputs = [o.split(".")[0] for o in outputs]

        return outputs

    def best(self, prompt: str) -> str:
        outputs = self._generator(prompt, max_length=256, num_beams=16)
        outputs = [g["generated_text"][len(prompt) :].strip() for g in outputs]  # type: ignore

        # Hack to just use the first sentence instead of the full output
        outputs = [o.split(".")[0] for o in outputs]

        return outputs[0]


class HuggingFaceLocalSummaryEngine(LMEngine):
    def __init__(self, model: str, device: Optional[str] = None):
        self._generator = pipeline("summarization", model=model, framework="pt", device=0)

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        if temperature is not None and temperature > 0:
            outputs = self._generator(
                prompt, max_length=256, num_return_sequences=n_completions, do_sample=True, temperature=temperature
            )
        else:
            outputs = self._generator(prompt, max_length=256, num_return_sequences=n_completions, do_sample=False)

        outputs = [g["summary_text"].strip() for g in outputs]  # type: ignore

        # Hack to just use the first sentence instead of the full output
        outputs = [o.split(".")[0] for o in outputs]

        return outputs

    def best(self, prompt: str) -> str:
        outputs = self._generator(prompt, max_length=256, num_beams=16)
        outputs = [g["summary_text"].strip() for g in outputs]  # type: ignore

        return outputs[0]


class GPTJ6B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-j-6B", device=device)


class GPT2(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2", device=device)


class GPT2Med(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-medium", device=device)


class GPT2Lg(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-large", device=device)


class GPT2XL(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-xl", device=device)


class DistilGPT2(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("distilgpt2", device=device)


class GPTNeo125M(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-125M", device=device)


class GPTNeo1B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-1.3B", device=device)


class GPTNeo2B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-2.7B", device=device)


class T5Base(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("t5-base", device=device)


class T5Small(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("t5-small", device=device)


class Pegasus(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("google/pegasus-xsum", device=device)
