from typing import Dict, Type

from .base import CaptionEngine
from .blip_engine import BLIP2COCOBase, BLIP2COCOLarge, BLIP2COCOT5Large, BLIPBase, BLIPLarge
from .ic3_engine import IC3CaptionEngine
from .ofa_engine import OFACaptionEngine
from .socratic_models import SocraticModelCaptionEngine

CAPTION_ENGINES: Dict[str, Type[CaptionEngine]] = {
    "BLIP (Large)": BLIPLarge,
    "BLIP (Base)": BLIPBase,
    "BLIP2 (OPT, COCO, 6.7B)": BLIP2COCOLarge,
    "BLIP2 (OPT, COCO, 2.7B)": BLIP2COCOBase,
    "BLIP2 (T5, COCO, flanT5XL)": BLIP2COCOT5Large,
    "OFA (Large + Caption)": OFACaptionEngine,
    "Socratic Models": SocraticModelCaptionEngine,
    "IC3": IC3CaptionEngine,
}

CAPTION_ENGINES_CLI: Dict[str, Type[CaptionEngine]] = {
    "blip": BLIPLarge,
    "blip-base": BLIPBase,
    "blip2": BLIP2COCOLarge,
    "blip2-base": BLIP2COCOBase,
    "blip2-t5": BLIP2COCOT5Large,
    "ofa": OFACaptionEngine,
    "socratic-models": SocraticModelCaptionEngine,
    "ic3": IC3CaptionEngine,
}
