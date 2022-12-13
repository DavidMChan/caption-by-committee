from .base import LMEngine  # noqa: F401
from .huggingface_inference_engine import Bloom, HuggingfaceInferenceLMEngine  # noqa: F401
from .huggingface_local_engine import HuggingFaceLocalLMEngine  # noqa: F401
from .huggingface_local_engine import HuggingFaceLocalSummaryEngine  # noqa: F401
from .huggingface_local_engine import (
    GPT2,
    GPT2XL,
    GPTJ6B,
    DistilGPT2,
    GPT2Lg,
    GPT2Med,
    GPTNeo1B,
    GPTNeo2B,
    GPTNeo125M,
    Pegasus,
    T5Base,
    T5Small,
)
from .openai_engine import GPT3Ada, GPT3Babbage, GPT3Curie, GPT3Davinci2, GPT3Davinci3, OpenAILMEngine  # noqa: F401

LM_ENGINES = {
    "GPT-3 (Davinci v3)": GPT3Davinci3,
    "GPT-3 (Davinci v2)": GPT3Davinci2,
    "GPT-3 (Curie)": GPT3Curie,
    "GPT-3 (Babbage)": GPT3Babbage,
    "GPT-3 (Ada)": GPT3Ada,
    "Bloom": Bloom,
    "GPT-Neo 125M": GPTNeo125M,
    "GPT-Neo 1.3B": GPTNeo1B,
    "GPT-Neo 2.7B": GPTNeo2B,
    "GPT-J-6B": GPTJ6B,
    "GPT-2": GPT2,
    "GPT-2 (Medium)": GPT2Med,
    "GPT-2 (Large)": GPT2Lg,
    "GPT-2 (XL)": GPT2XL,
    "DistilGPT-2": DistilGPT2,
    "T5 (Base)": T5Base,
    "T5 (Small)": T5Small,
    "Pegasus": Pegasus,
}
