import itertools
from typing import List, Optional

from rich.console import Console


def print_tokens_weighted(
    tokens: List[str], weights: List[float], strike: Optional[List[bool]] = None, direction="lower"
) -> None:
    # Normalize the weights between 0 and 1
    weights = [(w - min(weights)) / (max(weights) - min(weights)) for w in weights]

    # Print the tokens
    console = Console(color_system="truecolor")
    console.print("", end="", style="default on default")
    for t, w, s in zip(tokens, weights, strike if strike is not None else itertools.repeat(False)):
        console.print(
            f"[bold]{'[strike]' if s else ''}{t}{'[/strike]' if s else ''}[/bold]",
            style=f"on rgb({int((1 - w) * 255) if direction == 'lower' else int(w * 255)},0,0)",
            end=" ",
        )

    # Clear the formatting
    console.print(" \n", end="", style="default on default")
    console.print(style="default on default")
