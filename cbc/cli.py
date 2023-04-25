import click

from .caption_by_committee import caption
from .dataset import evaluate_dataset
from .plugins.test import test_plugin


@click.group()
def main() -> None:
    pass


main.add_command(evaluate_dataset)
main.add_command(caption)
main.add_command(test_plugin)
