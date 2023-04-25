import re
from typing import Callable, Dict

GUARDS: Dict[str, Callable[[str], bool]] = {}


def register_guard(func: Callable[[str], bool]) -> Callable[[str], bool]:
    # Register a guard function, which is a function that takes a generated caption, and returns a boolean which
    # indicates whether the caption should be kept or not.
    GUARDS[func.__name__] = func
    return func


@register_guard
def not_too_many_commas(caption: str) -> bool:
    # Lots of commas in a caption is an indicator that a list of objects was generated
    return caption.count(",") > 9  # noqa: PLR2004


@register_guard
def no_lists_of_tokens(caption: str) -> bool:
    # Lists of tokens are usually not useful. Match a regex that loos for things that go "x", "y", (and) "z"
    regex = re.compile(r'".*?"(,| and)".*?"(,| and)".*?"')
    return bool(regex.search(caption))


@register_guard
def no_mention_of_captions(caption: str) -> bool:
    # When the word "caption" or "captions" is in the caption, it's usually not useful
    return "caption" in caption.lower()


@register_guard
def no_mention_of_ocr(caption: str) -> bool:
    # When the word "ocr" or "ocrs" is in the caption, it's usually not useful
    return "ocr" in caption.lower()
