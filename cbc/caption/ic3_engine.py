import logging
from typing import List, Optional

import torch
from PIL import Image

from cbc.caption.base import CaptionEngine
from cbc.caption.ic3.caption_by_committee import DEFAULT_CBC_PROMPT, caption_by_committee


class IC3CaptionEngine(CaptionEngine):
    def __init__(
        self,
        caption_engine: str = "blip2",
        lm_engine: str = "stable_lm_7B",
        num_candidates: int = 10,
        candidate_temperature: float = 0.7,
        prompt: str = DEFAULT_CBC_PROMPT,
        plugin: List[str] = [],
        postprocessing: str = "default",
    ):
        from cbc.caption import CAPTION_ENGINES_CLI
        from cbc.lm import (
            LM_ENGINES_CLI,
            LM_LOCAL_ENGINES,
        )
        from cbc.plugins import IMAGE_PLUGINS

        logging.debug(f"Loading caption engine {caption_engine}...")
        self._captioner = CAPTION_ENGINES_CLI[caption_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

        logging.debug(f"Loading LM engine {lm_engine}...")
        if lm_engine in LM_LOCAL_ENGINES:
            self._lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
        else:
            self._lm = LM_ENGINES_CLI[lm_engine]()

        logging.debug(f'Loading plugins: {", ".join(plugin)}')
        self._plugins = [IMAGE_PLUGINS[p]() for p in plugin]

        self._num_candidates = num_candidates
        self._candidate_temperature = candidate_temperature
        self._prompt = prompt
        self._postprocessing = postprocessing

    def __call__(self, raw_image: Image.Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:
        logging.debug("Generating caption...")

        output_captions = caption_by_committee(
            raw_image.convert("RGB"),
            caption_engine=self._captioner,
            lm_engine=self._lm,
            caption_engine_temperature=self._candidate_temperature,
            n_captions=self._num_candidates,
            lm_prompt=self._prompt,
            postprocess=self._postprocessing,
            plugins=self._plugins,
            num_outputs=n_captions,
            lm_temperature=temperature or 1.0,
        )
        if isinstance(output_captions, str):
            output_captions = [output_captions]

        return output_captions

    def get_baseline_caption(self, raw_image: Image.Image) -> str:
        return self(raw_image, n_captions=1, temperature=1.0)[0]
