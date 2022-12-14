from typing import Dict, Set, Type

from .base import LMEngine  # noqa: F401
from .huggingface_inference_engine import Bloom  # noqa: F401
from .huggingface_inference_engine import HuggingfaceInferenceLMEngine
from .huggingface_local_engine import HuggingFaceLocalLMEngine  # noqa: F401
from .huggingface_local_engine import \
    HuggingFaceLocalSummaryEngine  # noqa: F401
from .huggingface_local_engine import (GPT2, GPT2XL, GPTJ6B, DistilGPT2,
                                       GPT2Lg, GPT2Med, GPTNeo1B, GPTNeo2B,
                                       GPTNeo125M, Pegasus, T5Base, T5Small)
from .openai_engine import (GPT3Ada, GPT3Babbage, GPT3Curie,  # noqa: F401
                            GPT3Davinci2, GPT3Davinci3, OpenAILMEngine)

LM_ENGINES: Dict[str, Type[LMEngine]] = {
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

LM_ENGINES_CLI: Dict[str, Type[LMEngine]] = {
    "gpt3_davinci3": GPT3Davinci3,
    "gpt3_davinci2": GPT3Davinci2,
    "gpt3_curie": GPT3Curie,
    "gpt3_babbage": GPT3Babbage,
    "gpt3_ada": GPT3Ada,
    "bloom": Bloom,
    "gpt_neo_125m": GPTNeo125M,
    "gpt_neo_1b": GPTNeo1B,
    "gpt_neo_2b": GPTNeo2B,
    "gpt_j_6b": GPTJ6B,
    "gpt2": GPT2,
    "gpt2_med": GPT2Med,
    "gpt2_lg": GPT2Lg,
    "gpt2_xl": GPT2XL,
    "distilgpt2": DistilGPT2,
    "t5_base": T5Base,
    "t5_small": T5Small,
    "pegasus": Pegasus,
}

LM_LOCAL_ENGINES: Set[str] = {
    "gpt_neo_125m",
    "gpt_neo_1b",
    "gpt_neo_2b",
    "gpt_j_6b",
    "gpt2",
    "gpt2_med",
    "gpt2_lg",
    "gpt2_xl",
    "distilgpt2",
    "t5_base",
    "t5_small",
    "pegasus",
}
