from typing import Dict, Type

from .base import CaptionEngine  # noqa: F401
from .blip_engine import BLIPBase, BLIPLarge
from .ofa_engine import OFACaptionEngine
from .socratic_models import SocraticModelCaptionEngine

CAPTION_ENGINES: Dict[str, Type[CaptionEngine]] = {
    "BLIP (Large)": BLIPLarge,
    "BLIP (Base)": BLIPBase,
    "OFA (Large + Caption)": OFACaptionEngine,
    "Socratic Models": SocraticModelCaptionEngine,
}

CAPTION_ENGINES_CLI: Dict[str, Type[CaptionEngine]] = {
    "blip": BLIPLarge,
    "blip-base": BLIPBase,
    "ofa": OFACaptionEngine,
    "socratic_models": SocraticModelCaptionEngine,
}
