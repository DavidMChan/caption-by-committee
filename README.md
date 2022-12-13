# Captioning by Comittee


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
```

Next, we need to set up environment variables with API keys, if you want to use those API keys
```bash
# For OpenAI-based models, specify the following keys:
export OPENAI_API_KEY=<api key>
export OPENAI_API_ORG=<org>

# For Huggingface Inference engine models, specify the following keys:
export HUGGINGFACE_API_KEY=<api key>
```



## Running the model using the CLI


## Running the model as an API


## Running the demos

To load the demos, install the library, and then use streamlit to run the demo:

*Single Image End-to-End Demo:* `streamlit run demos/single_image.py`
