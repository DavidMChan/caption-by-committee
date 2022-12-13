from .blip_engine import BLIPBase, BLIPLarge

CAPTION_ENGINES = {
    "BLIP (Large)": BLIPLarge,
    "BLIP (Base)": BLIPBase,
    "OFA (Large + Caption)": None,
    "Socratic Models": None,
}
