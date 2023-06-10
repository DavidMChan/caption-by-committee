import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import torch
from git.repo import Repo
from PIL import Image
from torchvision import transforms

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.utils.python import chdir
from cbc.utils.pytorch import select_device

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
            logging.debug(f"[Fetching OFA] git lfs pull output: {output!s}")
        except subprocess.CalledProcessError as e:
            logging.error(f"[Fetching OFA] git lfs pull failed: {e}")
            raise

    return cache_dir, cache_dir


class OFACaptionEngine(CaptionEngine):
    def __init__(
        self, ofa_model: str = "large-caption", device: Optional[str] = None, prompt: str = _OFA_DEFAULT_PROMPT
    ):
        device = select_device(device)
        tokenizer_path, model_path = _get_ofa_model(ofa_model)
        self.tokenizer = OFATokenizer.from_pretrained(tokenizer_path)
        self.model = OFAModel.from_pretrained(model_path, device=device, use_cache=True)
        self.model = self.model.to(device or "cpu").eval()
        self.prompt = prompt
        self.device = device

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
        postfix="",
    ) -> torch.Tensor:
        return self.tokenizer([self.prompt + postfix], return_tensors="pt").input_ids.to(self.device)

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
        patch_image = self._preprocess_image(raw_image).to(self.device)
        prefix = self._get_language_prompt().to(self.device)
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

    def likelihood(self, raw_image: Image, caption: str) -> Dict[str, torch.Tensor]:
        caption = " " + caption.lower().replace(
            ".", ""
        )  # Super weird quirk in how the captions are generated, requires a space at the beginning
        patch_img = self._preprocess_image(raw_image).to(self.device)
        inputs = self._get_language_prompt().to(self.device)
        decoder_input_ids = self.tokenizer(caption, return_tensors="pt").input_ids.to(self.device)

        # Prepare the inputs tensor
        inputs_tensor, _, model_kwargs = self.model._prepare_model_inputs(
            inputs,
            0,
            {
                "patch_images": patch_img,
            },
        )

        model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, None)
        outputs = self.model(decoder_input_ids=decoder_input_ids, **model_kwargs)  # type: ignore

        # Logits are outputs.logits, indices are decoder_input_ids. Shift the decoder input ids by one to get the
        # target indices, and then use those to index into the logits to get the log probabilities of the target
        logits_shifted = outputs.logits[:, :-1, :].contiguous()
        target_indices = decoder_input_ids[:, 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(logits_shifted, dim=-1)

        # Get the rank of the target indices in the log probabilities
        rank = torch.argsort(log_probs, dim=-1, descending=True)
        rank = torch.where(rank == target_indices.unsqueeze(-1))[2]
        # Print the average rank of the target indices
        logging.debug(f"Caption: {caption}, Average rank of target indices: {rank.float().mean().item()}")

        rank_score = (1 + rank) / log_probs.shape[-1]

        # Return the tensor of log probabilities
        outs = log_probs.gather(dim=-1, index=target_indices.unsqueeze(-1)).squeeze()

        # Get the top 5 tokens and their probabilities for each token in the caption
        topk = torch.topk(log_probs, k=5, dim=-1)

        # Do some cool printing of the output likelihoods
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
        output_topk_tokens = []
        for f, z, tk, r in zip(decoded_tokens[1:], outs, topk.indices[0], rank):
            # Deocde the top-k tokens
            topk_tokens = self.tokenizer.convert_ids_to_tokens(tk)
            topk_tokens = [t.replace("Ġ", "") for t in topk_tokens]  # Remove part separators
            output_topk_tokens.append(topk_tokens)

        return {
            "log_probs": outs,
            "ranks": rank,
            "normalized_rank_score": rank_score,
            "output_topk_tokens": output_topk_tokens,
            "output_tokens": decoded_tokens,
            "full_log_probs": log_probs,
        }
