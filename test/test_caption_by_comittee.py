from PIL import Image

from cbc.caption import OFACaptionEngine
from cbc.caption_by_committee import caption_by_committee
from cbc.lm import GPT3Davinci3


def test_caption_by_committee() -> None:
    image = Image.open("coco_test_images/COCO_val2014_000000165547.jpg").convert("RGB")
    caption_engine = OFACaptionEngine(ofa_model="/home/davidchan/Repos/OFA-large-caption/", device="cuda:1")
    lm_engine = GPT3Davinci3()
    caption = caption_by_committee(
        image,
        caption_engine=caption_engine,
        lm_engine=lm_engine,
        caption_engine_temperature=1.0,
        n_captions=15,
    )
    print(caption)


if __name__ == "__main__":
    test_caption_by_committee()
