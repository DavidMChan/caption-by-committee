from typing import Dict, List, Literal, Optional, Union

import torch
import transformers
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.blip_models.blip_caption import BlipCaption
from packaging import version
from PIL.Image import Image

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.utils.pytorch import select_device

BLIPModelType = Union[Literal["base_coco"], Literal["large_coco"]]
BLIP2Architecture = Union[
    Literal["blip2_opt"],
    Literal["blip2_t5"],
    Literal["blip2"],
]
BLIP2ModelType = Union[
    Literal["pretrain_opt2.7b"],
    Literal["caption_coco_opt2.7b"],
    Literal["pretrain_opt6.7b"],
    Literal["caption_coco_opt6.7b"],
    Literal["pretrain_flant5xl"],
    Literal["caption_coco_flant5xl"],
    Literal["pretrain_flant5xxl"],
    Literal["pretrain"],
    Literal["coco"],
]


def _generate_with_temperature(
    model: BlipCaption,
    samples: Dict[str, torch.Tensor],
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
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

    prompt = samples.get("prompt", model.prompt)
    if isinstance(prompt, str):
        prompt = [prompt] * image_embeds.size(0)

    prompt = model.tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore
    prompt.input_ids[:, 0] = model.tokenizer.bos_token_id  # type: ignore
    prompt.input_ids = prompt.input_ids[:, :-1]  # type: ignore

    # prepare prompt for beam search
    if not use_nucleus_sampling:
        prompt.input_ids = torch.repeat_interleave(prompt.input_ids, num_beams, dim=0)

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
    return [output[len(model.prompt) :] for output in outputs]


def _generate_opt_with_temperature(
    model: Blip2OPT,
    samples: Dict[str, torch.Tensor],
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
    max_length: int = 30,
    min_length: int = 10,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    num_captions: int = 1,
    temperature: float = 1.0,
) -> List[str]:
    # NOTE: This has to be overridden, since on it's own, lavis doesn't have a way to specify the temperature.

    image = samples["image"]
    with model.maybe_autocast():
        image_embeds = model.ln_vision(model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = model.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        prompt = samples.get("prompt", model.prompt)
        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(0), "The number of prompts must be equal to the batch size."

        opt_tokens = model.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # Weird bug here with generation...
        # It seems like this is broken in the old version of transformers
        if version.parse(transformers.__version__) >= version.parse("4.28.0"):
            query_embeds = inputs_opt
        else:
            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

        outputs = model.opt_model.generate(
            input_ids=input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            eos_token_id=model.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=1.0,
            num_return_sequences=num_captions,
        )

        prompt_length = opt_tokens.input_ids.shape[1]
        output_text = model.opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text


def _generate_t5_with_temperature(
    model: Blip2T5,
    samples: Dict[str, torch.Tensor],
    use_nucleus_sampling: bool = False,
    num_beams: int = 1,
    max_length: int = 30,
    min_length: int = 10,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    num_captions: int = 1,
    temperature: float = 1.0,
) -> List[str]:
    image = samples["image"]

    # Specify32 instead of bfloat16, since bfloat16 is not supported by a lot of GPUs
    with model.maybe_autocast(dtype=torch.float32):
        image_embeds = model.ln_vision(model.visual_encoder(image))

        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = model.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        prompt = samples.get("prompt", model.prompt)
        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(0), "The number of prompts must be equal to the batch size."

        input_tokens = model.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        inputs_embeds = model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = model.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=1.0,
            num_return_sequences=num_captions,
        )
        output_text = model.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_text


class BLIPCaptionEngine(CaptionEngine):
    def __init__(self, model: BLIPModelType = "large_coco", device: Optional[str] = None):
        device = select_device(device)
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
            output_generated = _generate_with_temperature(
                self._model,
                {"image": image},
                num_captions=n_captions,
                use_nucleus_sampling=True,
                top_p=0.9,
                temperature=temperature or 1.0,
            )
            output_captions += output_generated

        return [postprocess_caption(cap) for cap in output_captions]

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

    def get_ask_caption(self, raw_image: Image, question: str = "") -> str:
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore
        if question == "":
            samples = {"image": image}
        else:
            samples = {"image": image, "prompt": question}

        beam_search_caption = _generate_with_temperature(
            self._model,
            samples,
            num_captions=1,
            top_p=0.9,
            use_nucleus_sampling=False,
            num_beams=16,
        )

        return postprocess_caption(beam_search_caption[0])


class BLIP2CaptionEngine(CaptionEngine):
    def __init__(
        self,
        architecture: BLIP2Architecture = "blip2_opt",
        model: BLIP2ModelType = "caption_coco_opt6.7b",
        device: Optional[str] = None,
    ):
        device = select_device(device)
        self._model, self._vis_processors, _ = load_model_and_preprocess(
            name=architecture, model_type=model, device=device, is_eval=True
        )
        self._device = device
        self._architecture = architecture

    def __call__(self, raw_image: Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore

        output_captions = [self.get_baseline_caption(raw_image)]
        if n_captions > 1:
            n_captions -= 1  # We'll always add the baseline caption
            # Generate sampled captions
            if self._architecture == "blip2_opt":
                output_generated = _generate_opt_with_temperature(
                    self._model,
                    {"image": image},
                    num_captions=n_captions,
                    use_nucleus_sampling=True,
                    top_p=0.9,
                    temperature=temperature or 1.0,
                )
            elif self._architecture == "blip2_t5":
                output_generated = _generate_t5_with_temperature(
                    self._model,
                    {"image": image},
                    num_captions=n_captions,
                    use_nucleus_sampling=True,
                    top_p=0.9,
                    temperature=temperature or 1.0,
                )
            else:
                raise ValueError(f"Architecture {self._architecture} not supported for BLIP2CaptionEngine.")

            output_captions += output_generated

        return [postprocess_caption(cap) for cap in output_captions]

    def get_baseline_caption(self, raw_image: Image) -> str:
        # Generate best beam search caption
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore
        _gen_fn = (
            _generate_opt_with_temperature
            if self._architecture == "blip2_opt"
            else _generate_t5_with_temperature
            if self._architecture == "blip2_t5"
            else None
        )
        if _gen_fn is None:
            raise ValueError(f"Architecture {self._architecture} not supported for BLIP2CaptionEngine.")
        beam_search_caption = _gen_fn(
            self._model,
            {"image": image},
            num_captions=1,
            top_p=0.9,
            use_nucleus_sampling=False,
            num_beams=16,
        )

        return postprocess_caption(beam_search_caption[0])

    def get_ask_caption(self, raw_image: Image, question: str = "") -> str:
        # Generate best beam search caption
        image = self._vis_processors["eval"](raw_image).unsqueeze(0).to(self._device)  # type: ignore
        _gen_fn = (
            _generate_opt_with_temperature
            if self._architecture == "blip2_opt"
            else _generate_t5_with_temperature
            if self._architecture == "blip2_t5"
            else None
        )
        if _gen_fn is None:
            raise ValueError(f"Architecture {self._architecture} not supported for BLIP2CaptionEngine.")
        if question == "":
            samples = {"image": image}
        else:
            samples = {"image": image, "prompt": question}

        beam_search_caption = _gen_fn(
            self._model,
            samples,
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


class BLIP2COCOLarge(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(architecture="blip2_opt", model="caption_coco_opt6.7b", device=device)


class BLIP2COCOBase(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(architecture="blip2_opt", model="caption_coco_opt2.7b", device=device)


class BLIP2COCOT5Large(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(architecture="blip2_t5", model="caption_coco_flant5xl", device=device)


class BLIP2COCOT5XLarge(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None):
        super().__init__(architecture="blip2_t5", model="pretrain_flant5xxl", device=device)
