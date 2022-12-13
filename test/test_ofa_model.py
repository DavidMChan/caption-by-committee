from PIL import Image

from cbc.caption.ofa_engine import OFACaptionEngine


def test_ofa_base() -> None:
    engine = OFACaptionEngine(
        ofa_model="/home/davidchan/Repos/OFA-large-caption/", device="cuda:1"
    )  # Load using the default parameters

    # Load an image
    image = Image.open("coco_test_images/COCO_val2014_000000165547.jpg").convert("RGB")

    # Generate captions
    captions = engine(image, n_captions=5, temperature=1.0)

    # Print the captions
    for caption in captions:
        print(caption)


if __name__ == "__main__":
    test_ofa_base()
