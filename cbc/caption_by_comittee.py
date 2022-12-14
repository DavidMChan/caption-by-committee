# flake8: noqa

from typing import List

import click
import torch
from PIL import Image

from cbc.caption import CAPTION_ENGINES_CLI, CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.lm import LM_ENGINES_CLI, LM_LOCAL_ENGINES, LMEngine

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


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--caption-engine",
    type=click.Choice(CAPTION_ENGINES_CLI.keys()),  # type: ignore
    default="ofa",
    help="The underlying captioning model to use.",
)
@click.option(
    "--lm-engine",
    type=click.Choice(LM_ENGINES_CLI.keys()),  # type: ignore
    default="gpt3_davinci3",
    help="The LM to use.",
)
@click.option("--num-candidates", type=int, default=15, help="Number of candidates to generate for each image.")
@click.option("--candidate-temperature", type=float, default=1.0, help="Temperature to use when generating candidates.")
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_CBC_PROMPT,
    help="The prompt to use when generating candidates. Will load from a file if it exists.",
)
def caption(
    image_path: str,
    caption_engine: str,
    lm_engine: str,
    num_candidates: int,
    candidate_temperature: float,
    prompt: str,
) -> None:
    """
    Generate a caption for an image using a committee of captioning models.
    """
    image = Image.open(image_path).convert("RGB")
    print(f"Loading caption engine {caption_engine}...")
    captioner = CAPTION_ENGINES_CLI[caption_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    print(f"Loading LM engine {lm_engine}...")
    if lm_engine in LM_LOCAL_ENGINES:
        lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        lm = LM_ENGINES_CLI[lm_engine]()

    print("Generating caption...")
    summary = caption_by_comittee(
        image,
        caption_engine=captioner,
        lm_engine=lm,
        caption_engine_temperature=candidate_temperature,
        n_captions=num_candidates,
        lm_prompt=prompt,
    )

    print("Caption:", summary)
