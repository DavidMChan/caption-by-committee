from typing import Dict, Set, Type

from .bard_engine import BardEngine
from .base import LMEngine
from .huggingface_inference_engine import (  # noqa: F401
    OPT,
    Bloom,
    HuggingfaceInferenceLMEngine,
)
from .huggingface_llama_engine import (
    Alpaca7B,
    Koala7B,
    Koala13B_V1,
    Koala13B_V2,
    Llama7B,
    Llama13B,
    Llama30B,
    Llama65B,
    Vicuna_7B,
    Vicuna_13B,
)
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
    HuggingFaceLocalLMEngine,  # noqa: F401
    HuggingFaceLocalSummaryEngine,  # noqa: F401
    Pegasus,
    StableLM3B,
    StableLM7B,
    StableLMBase3B,
    StableLMBase7B,
    T5Base,
    T5Small,
)
from .openai_engine import (  # noqa: F401
    GPT4,
    GPT432K,
    ChatGPT,
    GPT3Ada,
    GPT3Babbage,
    GPT3Curie,
    GPT3Davinci2,
    GPT3Davinci3,
    OpenAI,
)
from .vertex_engine import PaLMEngine

LM_ENGINES: Dict[str, Type[LMEngine]] = {
    "ChatGPT": ChatGPT,
    "GPT-4": GPT4,
    "GPT-4 (32K)": GPT432K,
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
    "OPT (66B)": OPT,
    "LLama 7B": Llama7B,
    "LLama 13B": Llama13B,
    "LLama 30B": Llama30B,
    "LLama 65B": Llama65B,
    "Alpaca 7B": Alpaca7B,
    "Koala 7B": Koala7B,
    "Koala 13B V1": Koala13B_V1,
    "Koala 13B V2": Koala13B_V2,
    "Vicuna 7B": Vicuna_7B,
    "Vicuna 13B": Vicuna_13B,
    "Stable LM (3B)": StableLM3B,
    "Stable LM (7B)": StableLM7B,
    "Stable LM Base (3B)": StableLMBase3B,
    "Stable LM Base (7B)": StableLMBase7B,
    "Bard": BardEngine,
    "PaLM": PaLMEngine,
}

LM_ENGINES_CLI: Dict[str, Type[LMEngine]] = {
    "chatgpt": ChatGPT,
    "gpt4": GPT4,
    "gpt432k": GPT432K,
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
    "opt": OPT,
    "llama_7B": Llama7B,
    "llama_13B": Llama13B,
    "llama_30B": Llama30B,
    "llama_65B": Llama65B,
    "alpaca_7B": Alpaca7B,
    "koala_7B": Koala7B,
    "koala_13B_v1": Koala13B_V1,
    "koala_13B_v2": Koala13B_V2,
    "vicuna_7B": Vicuna_7B,
    "vicuna_13B": Vicuna_13B,
    "stable_lm_3B": StableLM3B,
    "stable_lm_7B": StableLM7B,
    "stable_lm_base_3B": StableLMBase3B,
    "stable_lm_base_7B": StableLMBase7B,
    "bard": BardEngine,
    "palm": PaLMEngine,
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
    "GPT-Neo 125M",
    "GPT-Neo 1.3B",
    "GPT-Neo 2.7B",
    "GPT-J-6B",
    "GPT-2",
    "GPT-2 (Medium)",
    "GPT-2 (Large)",
    "GPT-2 (XL)",
    "DistilGPT-2",
    "T5 (Base)",
    "T5 (Small)",
    "Pegasus",
    "llama_7B",
    "llama_13B",
    "llama_30B",
    "llama_65B",
    "alpaca_7B",
    "koala_7B",
    "koala_13B_v1",
    "koala_13B_v2",
    "vicuna_7B",
    "vicuna_13B",
    "stable_lm_3B",
    "stable_lm_7B",
    "stable_lm_base_3B",
    "stable_lm_base_7B",
}
