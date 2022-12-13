import os
from typing import Any, List, Optional, Tuple

import clip
import numpy as np
import torch
from PIL.Image import Image

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption

_DEFAULT_SM_PROMPT = """I am an intelligent image captioning bot.
    This image is a {img_type}. There {ppl_result}.
    I think this photo was taken at a {sorted_places[0]}, {sorted_places[1]}, or {sorted_places[2]}.
    I think there might be a {object_list} in this {img_type}.
    A creative short caption I can generate to describe this image is:"""

_DEFAULT_SM_IMAGE_TYPES = ["photo", "painting", "drawing", "illustration", "cartoon", "sketch", "painting", "drawing"]
_DEFAULT_SM_PEOPLE_TYPES = ["people", "no people"]
_DEFAULT_SM_PEOPLE_NUMBER_TYPES = [
    "is one person",
    "are two people",
    "are three people",
    "are several people",
    "are many people",
]
_DEFAULT_DM_PEOPLE_PROMPT = "There are {p} in this photo."


def _load_places_text() -> List[str]:
    place_categories = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "socratic_models_data", "categories_places365.txt"), dtype=str
    )
    place_texts = []

    for place in place_categories[:, 0]:
        place = place.split("/")[2:]
        if len(place) > 1:
            place = place[1] + " " + place[0]
        else:
            place = place[0]
        place = place.replace("_", " ")
        place_texts.append(place)

    return place_texts


def _load_object_category_text(place_texts: List[str]) -> List[str]:
    with open(
        os.path.join(os.path.dirname(__file__), "socratic_models_data", "dictionary_and_semantic_hierarchy.txt"),
        "r",
    ) as f:
        object_categories = f.readlines()
    object_texts = []
    for object_text in object_categories[1:]:
        object_text = object_text.strip()
        object_text = object_text.split("\t")[3]
        safe_list = ""
        for variant in object_text.split(","):
            text = variant.strip()
            safe_list += f"{text}, "
        safe_list = safe_list[:-2]
        if len(safe_list) > 0:
            object_texts.append(safe_list)

    ptx = set(place_texts)
    object_texts = [o for o in list(set(object_texts)) if o not in ptx]  # Remove redundant categories.

    return object_texts


class SocraticModelCaptionEngine(CaptionEngine):
    def __init__(
        self,
        language_model: Any = None,
        clip_version: str = "ViT-L/14",
        device: Optional[str] = None,
        prompt: str = _DEFAULT_SM_PROMPT,
        image_types: List[str] = _DEFAULT_SM_IMAGE_TYPES,
        people_types: List[str] = _DEFAULT_SM_PEOPLE_TYPES,
        people_number_types: List[str] = _DEFAULT_SM_PEOPLE_NUMBER_TYPES,
        people_prompt: str = _DEFAULT_DM_PEOPLE_PROMPT,
        top_k_objects: int = 10,
    ):
        # Initialize the CLIP model
        self._clip_model, self._clip_preprocess = clip.load(clip_version, device=device)  # type: ignore

        # Load the cached features
        self._place_texts = _load_places_text()
        self._object_texts = _load_object_category_text(self._place_texts)

        self._cached_place_features: Optional[torch.Tensor] = None
        self._cached_object_features: Optional[torch.Tensor] = None
        self._cached_people_number_features: Optional[torch.Tensor] = None
        self._cached_people_type_features: Optional[torch.Tensor] = None
        self._cached_image_type_features: Optional[torch.Tensor] = None

        # Store the prompts
        self._prompt = prompt
        self._image_types = image_types
        self._people_types = people_types
        self._people_number_types = people_number_types
        self._people_prompt = people_prompt
        self._top_k_objects = top_k_objects

        # Setup the language model
        self._language_model = language_model
        self._device = device

    def _get_image_features(self, image: Image) -> torch.Tensor:
        # Get the image features using the CLIP model
        image_in = self._clip_preprocess(image).unsqueeze(0).to(self._device)  # type: ignore
        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_in)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu()

        return image_features

    def _get_text_features(self, input_text: List[str], batch_size: int = 64) -> torch.Tensor:
        text_tokens = clip.tokenize(input_text).to(self._device)  # type: ignore
        text_id = 0

        output_text_features = []
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(input_text) - text_id, batch_size)
            text_batch = text_tokens[text_id : text_id + batch_size]
            with torch.no_grad():
                batch_feats = self._clip_model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = batch_feats.cpu()
            output_text_features.append(batch_feats)
            text_id += batch_size

        return torch.cat(output_text_features, dim=0)

    def _get_sorted_text_scores(
        self, raw_text: List[str], text_features: torch.Tensor, image_features: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        # Compute the similarity scores between the text and the image
        with torch.no_grad():
            text_image_scores = text_features.cpu().float() @ image_features.T.cpu().float()

        text_image_scores = text_image_scores.squeeze().float().numpy()
        high_to_low_ids = np.argsort(text_image_scores)[::-1]
        sorted_text = [raw_text[i] for i in high_to_low_ids]
        sorted_text_scores = [text_image_scores[i] for i in high_to_low_ids]
        return sorted_text, sorted_text_scores  # type: ignore

    def _get_prompt(self, raw_image: Image) -> str:
        # Generate captions using the Socratic Model
        image_features = self._get_image_features(raw_image)

        # Handle the caching of the features
        if self._cached_place_features is None:
            self._cached_place_features = self._get_text_features(self._place_texts)
        if self._cached_object_features is None:
            self._cached_object_features = self._get_text_features(self._object_texts)

        # Classify the image type
        if self._cached_image_type_features is None:
            self._cached_image_type_features = self._get_text_features([f"This is a {t}." for t in self._image_types])
        sorted_image_types, _ = self._get_sorted_text_scores(
            self._image_types, self._cached_image_type_features, image_features
        )
        output_image_type = sorted_image_types[0]

        # Classify the number of people
        if self._cached_people_type_features is None:
            self._cached_people_type_features = self._get_text_features(
                [f"There are {t} people." for t in self._people_types]
            )
        sorted_people_types, _ = self._get_sorted_text_scores(
            self._people_types, self._cached_people_type_features, image_features
        )
        output_people_type = f"are {sorted_people_types[0]}"
        if sorted_people_types[0] == "people":
            if self._cached_people_number_features is None:
                self._cached_people_number_features = self._get_text_features(
                    [f"There {t} in this photo." for t in self._people_number_types]
                )
            sorted_people_number_types, _ = self._get_sorted_text_scores(
                self._people_number_types, self._cached_people_number_features, image_features
            )
            output_people_type = sorted_people_number_types[0]

        # Classify the places
        sorted_places, _ = self._get_sorted_text_scores(self._place_texts, self._cached_place_features, image_features)

        # Classify the objects
        sorted_objects, _ = self._get_sorted_text_scores(
            self._object_texts, self._cached_object_features, image_features
        )
        output_object_list = ", ".join(sorted_objects[: self._top_k_objects])

        # Generate the prompt
        prompt = self._prompt.format(
            img_type=output_image_type,
            ppl_result=output_people_type,
            sorted_places=sorted_places,
            object_list=output_object_list,
        )

        return prompt

    def __call__(self, raw_image: Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:
        prompt = self._get_prompt(raw_image)
        print(prompt)
        # Generate captions using the language model
        return [
            postprocess_caption(p) for p in self._language_model(prompt, n_capitons=n_captions, temperature=temperature)
        ]

    def get_baseline_caption(self, raw_image: Image) -> str:
        # Generate baseline caption using the Socratic Model
        prompt = self._get_prompt(raw_image)
        return self._language_model.best(prompt)
