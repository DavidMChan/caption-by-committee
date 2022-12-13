from PIL import Image

from cbc.caption.blip_engine import BLIPBase, BLIPLarge


def test_blip_base():
    engine = BLIPBase(device="cuda:1")  # Load using the default parameters

    # Load an image
    image = Image.open("coco_test_images/COCO_val2014_000000165547.jpg").convert("RGB")

    # Generate captions
    captions = engine(image, n_captions=5, temperature=1.0)

    # Print the captions
    for caption in captions:
        print(caption)


if __name__ == "__main__":
    test_blip_base()
