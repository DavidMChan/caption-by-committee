import logging
import os
import subprocess
from typing import List, Optional, Tuple

import torch
from git.repo import Repo
from PIL import Image
from torchvision import transforms

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.utils.python import chdir

from .ofa import OFAModel, OFATokenizer

_OFA_PATHS = {
    "large-caption": "https://huggingface.co/OFA-Sys/OFA-large-caption",
}

_OFA_DEFAULT_PROMPT = " what does the image describe?"


def _get_ofa_model(model: str) -> Tuple[str, str]:
    if os.path.exists(model):
        # We can load the model locally
        return model, model

    if model not in _OFA_PATHS:
        raise ValueError(f"Invalid OFA model: {model}, should be one of {list(_OFA_PATHS.keys())}")

    git_repo = _OFA_PATHS[model]

    # If the repo is already cloned, we can use it
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "cbc", model)
    if os.path.exists(cache_dir):
        # Check to see if the checkpoint files are there...
        # Check if the size > 1.8GB
        if os.path.exists(os.path.join(cache_dir, "pytorch_model.bin")):
            if os.path.getsize(os.path.join(cache_dir, "pytorch_model.bin")) > 1.8e9:  # noqa: PLR2004
                logging.debug(f"[Fetching OFA] Using cached model at {cache_dir}")
                return cache_dir, cache_dir
    else:
        logging.debug(f"[Fetching OFA] Model not found at {cache_dir}, cloning from {git_repo}")

    # Clone the repo into a cache directory
    os.makedirs(cache_dir, exist_ok=True)
    repo = Repo.clone_from(git_repo, cache_dir, branch="main")
    repo.git.checkout("main")

    # Run git lfs pull to download the model files
    with chdir(cache_dir):
        logging.debug(f"[Fetching OFA] Running git lfs pull in {cache_dir}...")
        try:
            output = subprocess.check_output(["git", "lfs", "pull"])
            logging.debug(f"[Fetching OFA] git lfs pull output: {str(output)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"[Fetching OFA] git lfs pull failed: {e}")
            raise

    return cache_dir, cache_dir


class OFACaptionEngine(CaptionEngine):
    def __init__(
        self, ofa_model: str = "large-caption", device: Optional[str] = None, prompt: str = _OFA_DEFAULT_PROMPT
    ):

        tokenizer_path, model_path = _get_ofa_model(ofa_model)
        self.tokenizer = OFATokenizer.from_pretrained(tokenizer_path)
        self.model = OFAModel.from_pretrained(model_path, device=device, use_cache=True)
        self.model = self.model.to(device or "cpu").eval()
        self.prompt = prompt
        self.device = device or "cpu"

    def _preprocess_image(self, raw_image: Image.Image) -> torch.Tensor:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        patch_resize_transform = transforms.Compose(
            [
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),  # type: ignore
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return patch_resize_transform(raw_image).unsqueeze(0).to(self.device)  # type: ignore

    def _get_language_prompt(
        self,
    ) -> torch.Tensor:
        return self.tokenizer([self.prompt], return_tensors="pt").input_ids.to(self.device)

    def __call__(self, raw_image: Image.Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:

        patch_img = self._preprocess_image(raw_image)
        inputs = self._get_language_prompt()

        output_captions = [self.get_baseline_caption(raw_image)]
        if n_captions > 1:
            for _ in range(n_captions - 1):
                # Sample from the model
                gen = self.model.generate(  # type: ignore
                    inputs,
                    patch_images=patch_img,
                    do_sample=True,
                    top_p=0.9,
                    temperature=temperature,
                    no_repeat_ngram_size=3,
                    max_length=256,
                    num_beams=1,
                )
                output_captions.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])

        return [postprocess_caption(caption.strip()) for caption in output_captions]

    def get_baseline_caption(self, raw_image: Image.Image) -> str:
        patch_image = self._preprocess_image(raw_image)
        prefix = self._get_language_prompt()
        baseline_gen = self.model.generate(  # type: ignore
            prefix,
            patch_images=patch_image,
            num_beams=16,
            no_repeat_ngram_size=3,
            max_length=256,
        )
        baseline_caption = self.tokenizer.batch_decode(baseline_gen, skip_special_tokens=True)
        baseline_caption = postprocess_caption(baseline_caption[0].strip())

        return baseline_caption
