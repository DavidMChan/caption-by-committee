from PIL import Image

from cbc.caption import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.lm import LMEngine

DEFAULT_CBC_PROMPT = """This is a hard problem. Carefully summarize in ONE detailed sentence the following captions by different (possibly incorrect) people describing the same thing. Be sure to describe everything, and identify when you're not sure. For example:
Captions: {}.
Summary:  I'm not sure, but the image is likely of"""


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
    # Post-process captions
    captions = [postprocess_caption(c) for c in captions]
    captions_formatted = [f'"{c}"' for c in captions]

    # Generate the prompt
    prompt = lm_prompt.format(", ".join(captions_formatted))

    # Generate the summary
    summary = lm_engine.best(prompt)

    return postprocess_caption(summary)
