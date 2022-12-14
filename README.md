# Captioning by Comittee

This is the implementation of the paper: *PAPER TITLE TO COME*.


## Installation

The library can be installed with:
```bash
# We can't install CLIP with setuptools, so we have to install it on its own
$ pip install git+https://github.com/openai/CLIP.git
# We also can't install LAVIS with setuptools, so we have to install it on its own
# We specify no-deps here, since LAVIS doesn't like new versions of pytorch (even though it supports it)
$ pip install git+https://github.com/salesforce/LAVIS.git --no-deps
# Install the local directory with setuptools
$ pip install .
# For the metrics, we need to download and install a spacy model
$ python -m spacy download en_core_web_lg
```

Next, we need to set up environment variables with API keys, if you want to use those API keys
```bash
# For OpenAI-based models, specify the following keys:
export OPENAI_API_KEY=<api key>
export OPENAI_API_ORG=<org>

# For Huggingface Inference engine models, specify the following keys:
export HUGGINGFACE_API_KEY=<api key>
```

The repository can be tested by running `cbc caption test/test_image.jpg`, which should produce a sample caption using
the OFA and GPT-2 models.

## Running the model using the CLI

To run the model using the CLI, you can use:
```bash
$ cbc caption <image path>
```

If you have a full dataset of examples, you can use:
```bash
$ cbc evaluate-dataset <dataset json>
```

Where the JSON format (minimally) looks like:
```json
[
    {
        "references": ["List", "of", "references"],
        "image_path": "Relative path to image"
    },
    ...
]
```

For more details on these commands, see `cbc caption --help` and `cbc evalaute --help`.


## Running the model as an REST API

*COMING SOON*

## Using the python API

To use the python API, see the following minimal example using GPT3 and OFA:

```python
from cbc.caption import OFACaptionEngine
from cbc.caption_by_comittee import caption_by_comittee
from cbc.lm import GPT3Davinci3

def run_caption() -> None:
    # Load the image
    image = Image.open("coco_test_images/COCO_val2014_000000165547.jpg").convert("RGB")

    # Construct a captioning engine (see: cbc/caption/__init__.py for available engines)
    caption_engine = OFACaptionEngine(device="cuda:1")

    # Construct a language model engine (see cbc/lm/__init__.py for available engines)
    lm_engine = GPT3Davinci3()

    # Generate the caption
    caption = caption_by_comittee(
        image,
        caption_engine=caption_engine,
        lm_engine=lm_engine,
        caption_engine_temperature=1.0,
        n_captions=15,
    )

    print(caption)

```


## Running the demos

To load the demos, install the library, and then use streamlit to run the demo:

*Single Image End-to-End Demo:* `streamlit run demos/single_image.py`


## References

If you found this work useful, cite us:
```
{

}
```
