import click

from .caption_by_committee import caption, video
from .dataset import evaluate_dataset


@click.group()
def main() -> None:
    pass


main.add_command(evaluate_dataset)
main.add_command(caption)
main.add_command(video)
