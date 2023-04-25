# flake8: noqa

from typing import List

import click
import torch
from PIL import Image
import logging

from cbc.caption import CAPTION_ENGINES_CLI
from cbc.lm import (
    LM_ENGINES_CLI,
    LM_LOCAL_ENGINES,
    OpenAI,
)
from cbc.plugins import IMAGE_PLUGINS

from rich.logging import RichHandler
from cbc.caption.ic3.caption_by_committee import DEFAULT_CBC_PROMPT, caption_by_committee


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--caption-engine",
    type=click.Choice(CAPTION_ENGINES_CLI.keys()),  # type: ignore
    default="blip2",
    help="The underlying captioning model to use.",
)
@click.option(
    "--lm-engine",
    type=click.Choice(LM_ENGINES_CLI.keys()),  # type: ignore
    default="stable_lm_7B",
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
@click.option(
    "--plugin",
    "-p",
    type=click.Choice(IMAGE_PLUGINS.keys()),  # type: ignore
    multiple=True,
    default=[],
    help="Plugins to use. Can be specified multiple times.",
)
@click.option(
    "--postprocessing",
    "-pp",
    type=click.Choice(["default", "all", "all_truncate"]),
    default="default",
    help="Whether to apply postprocessing to the final caption.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Whether to print verbose output.",
)
def caption(
    image_path: str,
    caption_engine: str,
    lm_engine: str,
    num_candidates: int,
    candidate_temperature: float,
    prompt: str,
    plugin: List[str],
    postprocessing: str,
    verbose: bool,
) -> None:
    """
    Generate a caption for an image using a committee of captioning models.
    """

    # Setup rich logging
    logging.basicConfig(
        level="DEBUG" if verbose else "INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )

    # Capture warnings
    logging.captureWarnings(True)

    image = Image.open(image_path).convert("RGB")
    logging.debug(f"Loading caption engine {caption_engine}...")
    captioner = CAPTION_ENGINES_CLI[caption_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    logging.debug(f"Loading LM engine {lm_engine}...")
    if lm_engine in LM_LOCAL_ENGINES:
        lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        lm = LM_ENGINES_CLI[lm_engine]()

    logging.debug(f'Loading plugins: {", ".join(plugin)}')
    plugins = [IMAGE_PLUGINS[p]() for p in plugin]

    logging.debug("Generating caption...")
    summary = caption_by_committee(
        image,
        caption_engine=captioner,
        lm_engine=lm,
        caption_engine_temperature=candidate_temperature,
        n_captions=num_candidates,
        lm_prompt=prompt,
        postprocess=postprocessing,
        plugins=plugins,
    )

    logging.debug(f"Caption: {summary}")
    if OpenAI.USAGE > 0:
        logging.debug(f"Total OpenAI API Usage Cost: ${OpenAI.USAGE:.6f}")

    print(summary)
