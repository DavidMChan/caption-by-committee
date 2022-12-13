from PIL import Image

from cbc.caption.socratic_models import SocraticModelCaptionEngine


def test_socratic_models() -> None:
    engine = SocraticModelCaptionEngine(device="cuda:1")  # Load using the default parameters

    # Load an image
    image = Image.open("coco_test_images/COCO_val2014_000000165547.jpg").convert("RGB")

    # Generate captions
    captions = engine(image, n_captions=5, temperature=1.0)

    # Print the captions
    for caption in captions:
        print(caption)


if __name__ == "__main__":
    test_socratic_models()
