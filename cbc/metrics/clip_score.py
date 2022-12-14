import os
from typing import Any, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
import tqdm
from PIL import Image

_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def clip_model() -> Tuple[Any, Any, str]:
    global _CLIP_MODEL
    global _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-L/14", device="cuda:0" if torch.cuda.is_available() else "cpu")
    return _CLIP_MODEL, _CLIP_PREPROCESS, "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_feature(media_path: str) -> torch.Tensor:
    model, preprocess, device = clip_model()
    image = preprocess(Image.open(media_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.reshape(-1)


def _get_image_feature_db(
    samples: List[Dict[str, Any]], image_path_key: str, image_root: Optional[str] = None
) -> torch.Tensor:
    features = []
    for sample in samples:
        media_path = os.path.join(image_root or "", sample[image_path_key])
        features.append(_get_feature(media_path))
    return torch.stack(features).to("cpu" if not torch.cuda.is_available() else "cuda:0")


def _get_text_features(
    candidates: List[str], references: List[str], baselines: List[str], char_limit: int = 300
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model, _, device = clip_model()

    while True:
        try:
            candidate_text = clip.tokenize([i[:char_limit] for i in candidates]).to(device)  # type: ignore
            reference_text = clip.tokenize([i[:char_limit] for i in references]).to(device)  # type: ignore
            baseline_text = clip.tokenize([i[:char_limit] for i in baselines]).to(device)  # type: ignore
            break
        except RuntimeError:
            # Back off the character limit
            if char_limit < 20:
                raise RuntimeError("Could not tokenize text -- too long?")
            char_limit -= 20

    with torch.no_grad():
        candidate_text_features = model.encode_text(candidate_text)
        reference_text_features = model.encode_text(reference_text)
        baseline_text_features = model.encode_text(baseline_text)
        candidate_text_features /= candidate_text_features.norm(dim=-1, keepdim=True)
        reference_text_features /= reference_text_features.norm(dim=-1, keepdim=True)
        baseline_text_features /= baseline_text_features.norm(dim=-1, keepdim=True)

    return candidate_text_features, reference_text_features, baseline_text_features


def compute_clips(
    index: int, feature_db: torch.Tensor, candidate: str, reference: str, baseline: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_features, reference_features, baseline_features = _get_text_features([candidate], [reference], [baseline])
    candidate_similarity_scores = feature_db @ candidate_features.T
    candidate_ranks = (candidate_similarity_scores > candidate_similarity_scores[index]).sum(dim=0)

    reference_similarity_scores = feature_db @ reference_features.T
    reference_ranks = (reference_similarity_scores > reference_similarity_scores[index]).sum(dim=0)

    baseline_similarity_scores = feature_db @ baseline_features.T
    baseline_ranks = (baseline_similarity_scores > baseline_similarity_scores[index]).sum(dim=0)

    return (candidate_ranks + 1).cpu().numpy(), (reference_ranks + 1).cpu().numpy(), (baseline_ranks + 1).cpu().numpy()


def compute_and_add_clip_recall(
    samples: List[Dict[str, Any]], image_path_key: str, image_root: Optional[str] = None
) -> List[Dict[str, Any]]:

    feature_db = _get_image_feature_db(samples, image_path_key, image_root)

    for index, sample in enumerate(tqdm.tqdm(samples)):
        candidate_ranks, reference_ranks, baseline_ranks = compute_clips(
            index, feature_db, sample["candidate_summary"], sample["reference_summary"], sample["baseline"]
        )

        if "scores" not in sample:
            sample["scores"] = {}

        # Mean Rank
        sample["scores"]["candidate_summary_clip_recall_rank"] = float(np.mean(candidate_ranks))
        sample["scores"]["reference_summary_clip_recall_rank"] = float(np.mean(reference_ranks))
        sample["scores"]["baseline_clip_recall_rank"] = float(np.mean(baseline_ranks))

        # Mean Reciprocal Rank
        sample["scores"]["candidate_summary_clip_recall_mrr"] = float(np.mean(1 / candidate_ranks))
        sample["scores"]["reference_summary_clip_recall_mrr"] = float(np.mean(1 / reference_ranks))
        sample["scores"]["baseline_clip_recall_mrr"] = float(np.mean(1 / baseline_ranks))

        # Mean Recall @ 1
        sample["scores"]["candidate_summary_clip_recall_at_1"] = float(np.mean(candidate_ranks <= 1))
        sample["scores"]["reference_summary_clip_recall_at_1"] = float(np.mean(reference_ranks <= 1))
        sample["scores"]["baseline_clip_recall_at_1"] = float(np.mean(baseline_ranks <= 1))

        # Mean Recall @ 5
        sample["scores"]["candidate_summary_clip_recall_at_5"] = float(np.mean(candidate_ranks <= 5))
        sample["scores"]["reference_summary_clip_recall_at_5"] = float(np.mean(reference_ranks <= 5))
        sample["scores"]["baseline_clip_recall_at_5"] = float(np.mean(baseline_ranks <= 5))

        # Mean Recall @ 10
        sample["scores"]["candidate_summary_clip_recall_at_10"] = float(np.mean(candidate_ranks <= 10))
        sample["scores"]["reference_summary_clip_recall_at_10"] = float(np.mean(reference_ranks <= 10))
        sample["scores"]["baseline_clip_recall_at_10"] = float(np.mean(baseline_ranks <= 10))

        # Max Recall Rank
        sample["scores"]["candidate_summary_clip_recall_max_rank"] = float(np.amax(candidate_ranks))
        sample["scores"]["reference_summary_clip_recall_max_rank"] = float(np.amax(reference_ranks))
        sample["scores"]["baseline_clip_recall_max_rank"] = float(np.amax(baseline_ranks))

    return samples
