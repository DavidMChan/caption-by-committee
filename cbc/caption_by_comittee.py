# flake8: noqa

from typing import List

from PIL import Image

from cbc.caption import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.lm import LMEngine

DEFAULT_CBC_PROMPT = """This is a hard problem. Carefully summarize in ONE detailed sentence the following captions by different (possibly incorrect) people describing the same thing. Be sure to describe everything, and identify when you're not sure. For example:
Captions: {}.
Summary:  I'm not sure, but the image is likely of"""


def get_prompt_for_candidates(candidates: List[str], prompt: str = DEFAULT_CBC_PROMPT) -> str:
    """
    Generate a prompt for a list of candidates.
    """
    candidates = [postprocess_caption(c) for c in candidates]
    candidates_formatted = [f'"{c}"' for c in candidates]
    return prompt.format(", ".join(candidates_formatted))


def caption_by_comittee(
    raw_image: Image.Image,
    caption_engine: CaptionEngine,
    lm_engine: LMEngine,
    caption_engine_temperature: float = 1.0,
    n_captions: int = 5,
    lm_prompt: str = DEFAULT_CBC_PROMPT,
) -> str:

    """
    Generate a caption for an image using a committee of captioning models.
    """

    captions = caption_engine(raw_image, n_captions=n_captions, temperature=caption_engine_temperature)
    prompt = get_prompt_for_candidates(captions, prompt=lm_prompt)
    summary = lm_engine.best(prompt)
    return postprocess_caption(summary)
