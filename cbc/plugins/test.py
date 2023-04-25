import logging
from typing import List

import click
from PIL import Image
from rich.logging import RichHandler

from cbc.lm import OpenAI
from cbc.plugins import IMAGE_PLUGINS


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--plugin",
    "-p",
    type=click.Choice(IMAGE_PLUGINS.keys()),  # type: ignore
    multiple=True,
    default=[],
    help="Plugins to use. Can be specified multiple times.",
)
def test_plugin(
    image_path: str,
    plugin: List[str],
) -> None:
    """
    Generate a caption for an image using a committee of captioning models.
    """

    # Setup rich logging
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=True,
    )

    # Capture warnings
    logging.captureWarnings(True)

    image = Image.open(image_path).convert("RGB")

    for p in plugin:
        logging.info("Testing plugin: " + p)
        plugin = IMAGE_PLUGINS[p]()
        plugin_outputs = plugin(image)
        logging.info(plugin_outputs)

    # print the OpenAI usage if we're using it
    if OpenAI.USAGE > 0:
        logging.info(f"Total OpenAI API Usage Cost: ${OpenAI.USAGE:.6f}")
