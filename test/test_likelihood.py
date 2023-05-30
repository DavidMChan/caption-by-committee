import os

from PIL import Image

from cbc.caption import OFACaptionEngine


def test_ofa_likelihood() -> None:

    # Load the test image
    image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    image = Image.open(image_path)
    caption_good = "A vase of sunflowers on a table."
    caption_bad = "A photo of spiderman and batman."

    # Load the caption engine
    caption_engine = OFACaptionEngine(device="cuda")

    caption_engine.get_baseline_caption(image)

    # Check the likelihood of the good caption
    likelihood_good = caption_engine.normalized_likelihood(image, caption_good)
    likelihood_bad = caption_engine.normalized_likelihood(image, caption_bad)

    print(f"Likelihood of good caption: {likelihood_good}")
    print(f"Likelihood of bad caption: {likelihood_bad}")

    # Check that the good caption has a higher likelihood than the bad caption
    assert likelihood_good > likelihood_bad


if __name__ == "__main__":
    test_ofa_likelihood()
