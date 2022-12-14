from typing import Dict, List, Literal, Optional, Union

import torch
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_caption import BlipCaption
from PIL.Image import Image

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption

BLIPModelType = Union[Literal["base_coco"], Literal["large_coco"]]


def _generate_with_temperature(
    model: BlipCaption,
    samples: Dict[str, torch.Tensor],
    use_nucleus_sampling: bool = False,
    num_beams: int = 3,
    max_length: int = 30,
    min_length: int = 10,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    num_captions: int = 1,
    temperature: float = 1.0,
) -> List[str]:
    # NOTE: This has to be overridden, since on it's own, lavis doesn't have a way to specify the temperature.

    # prepare inputs for decoder generation.
    encoder_out = model.forward_encoder(samples)  # type: ignore
    image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

    prompt = [model.prompt] * image_embeds.size(0)
    prompt = model.tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore
    prompt.input_ids[:, 0] = model.tokenizer.bos_token_id  # type: ignore
    prompt.input_ids = prompt.input_ids[:, :-1]  # type: ignore

    # get decoded text
    decoder_out = model.text_decoder.generate_from_encoder(  # type: ignore
        tokenized_prompt=prompt,
        visual_embeds=image_embeds,
        sep_token_id=model.tokenizer.sep_token_id,  # type: ignore
        pad_token_id=model.tokenizer.pad_token_id,  # type: ignore
        use_nucleus_sampling=use_nucleus_sampling,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
    )  # type: ignore

    outputs = model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)  # type: ignore
    captions = [output[len(model.prompt) :] for output in outputs]  # type: ignore

    return captions


class BLIPCaptionEngine(CaptionEngine):
    def __init__(self, model: BLIPModelType = "large_coco", device: Optional[str] = None):
        self._model, self._vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type=model, device=device, is_eval=True  # type: ignore
        )
        self._device = device

    def __call__(self, raw_image: Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore

        output_captions = [self.get_baseline_caption(raw_image)]
        if n_captions > 1:
            n_captions -= 1  # We'll always add the baseline caption
            # Generate sampled captions
            output_captions = _generate_with_temperature(
                self._model,
                {"image": image},
                num_captions=n_captions,
                use_nucleus_sampling=True,
                top_p=0.9,
                temperature=temperature or 1.0,
            )
            output_captions = [postprocess_caption(cap) for cap in output_captions]

        return output_captions

    def get_baseline_caption(self, raw_image: Image) -> str:
        # Generate best beam search caption
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore
        beam_search_caption = _generate_with_temperature(
            self._model,
            {"image": image},
            num_captions=1,
            top_p=0.9,
            use_nucleus_sampling=False,
            num_beams=16,
        )

        return postprocess_caption(beam_search_caption[0])


class BLIPLarge(BLIPCaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(model="large_coco", device=device)


class BLIPBase(BLIPCaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(model="base_coco", device=device)
