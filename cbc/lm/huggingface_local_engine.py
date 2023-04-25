from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.pipelines import pipeline

from cbc.lm.base import LMEngine
from cbc.utils.python import singleton
from cbc.utils.pytorch import select_device


class HuggingFaceLocalLMEngine(LMEngine):
    def __init__(self, model: str, device: Optional[str] = None):
        pipeline_device = select_device(device)
        self._generator = pipeline(
            "text-generation",
            model=model,
            framework="pt",
            device=pipeline_device if pipeline_device != "cpu" else None,
        )

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        if temperature is not None and temperature > 0:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
            )
        elif n_completions > 1:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=True,
                return_full_text=False,
            )
        else:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=False,
                return_full_text=False,
            )

        outputs = [g["generated_text"].strip() for g in outputs]  # type: ignore

        return outputs

    def best(self, prompt: str) -> str:
        outputs = self._generator(prompt, max_new_tokens=256, min_new_tokens=10, num_beams=16, return_full_text=False)
        outputs = [g["generated_text"].strip() for g in outputs]  # type: ignore

        return outputs[0]


class HuggingFaceLocalSummaryEngine(LMEngine):
    def __init__(self, model: str, device: Optional[str] = None):
        pipeline_device = select_device(device)
        self._generator = pipeline(
            "summarization", model=model, framework="pt", device=pipeline_device if pipeline_device != "cpu" else None
        )

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        if temperature is not None and temperature > 0:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
            )
        else:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=False,
                return_full_text=False,
            )

        outputs = [g["summary_text"].strip() for g in outputs]  # type: ignore

        return outputs

    def best(self, prompt: str) -> str:
        outputs = self._generator(prompt, max_new_tokens=256, min_new_tokens=10, num_beams=16, return_full_text=False)
        outputs = [g["summary_text"].strip() for g in outputs]  # type: ignore

        return outputs[0]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StableLMChatEngine(LMEngine):

    SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human."""

    def __init__(self, model: str, device: Optional[str] = None):
        device = select_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self._model.half().to(device)
        self._device = device

    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        output = self._tokenizer.decode(tokens).strip()
        _, _, output = output.partition("<|ASSISTANT|>")
        output = (
            output.replace("<|USER|>", "")
            .replace("<|ASSISTANT|>", "")
            .replace("<|SYSTEM|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        return output

    def __call__(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:

        # Filter the prompt (one-off experiment)
        prompt = prompt.replace("Summary:", "\nComplete the following sentence according to the task above:\nSummary: ")
        prompt = f"{StableLMChatEngine.SYSTEM_PROMPT}<|USER|>{prompt}<|ASSISTANT|>"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        if temperature is not None and temperature > 0:
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=5,
                temperature=temperature,
                num_return_sequences=n_completions,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )
        else:
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=5,
                num_return_sequences=n_completions,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )

        outputs = [self._decode_tokens(t) for t in tokens]  # type: ignore
        return outputs

    def best(self, prompt: str) -> str:

        with torch.no_grad():

            pre_input_prompt = prompt.replace("Summary:", "\nComplete the following sentence:\n")
            lm_prompt = f"{StableLMChatEngine.SYSTEM_PROMPT}<|USER|>{pre_input_prompt}<|ASSISTANT|>"
            inputs = self._tokenizer(lm_prompt, return_tensors="pt").to(self._device)

            tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=5,
                num_beams=16,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )

        return self._decode_tokens(tokens[0])


@singleton
class GPTJ6B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-j-6B", device=device)


@singleton
class GPT2(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2", device=device)


@singleton
class GPT2Med(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-medium", device=device)


@singleton
class GPT2Lg(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-large", device=device)


@singleton
class GPT2XL(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-xl", device=device)


@singleton
class DistilGPT2(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("distilgpt2", device=device)


@singleton
class GPTNeo125M(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-125M", device=device)


@singleton
class GPTNeo1B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-1.3B", device=device)


@singleton
class GPTNeo2B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-2.7B", device=device)


@singleton
class StableLMBase3B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-base-alpha-3b", device=device)


@singleton
class StableLMBase7B(HuggingFaceLocalLMEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-base-alpha-7b", device=device)


@singleton
class T5Base(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("t5-base", device=device)


@singleton
class T5Small(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("t5-small", device=device)


@singleton
class Pegasus(HuggingFaceLocalSummaryEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("google/pegasus-xsum", device=device)


@singleton
class StableLM3B(StableLMChatEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-tuned-alpha-3b", device=device)


@singleton
class StableLM7B(StableLMChatEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-tuned-alpha-7b", device=device)
