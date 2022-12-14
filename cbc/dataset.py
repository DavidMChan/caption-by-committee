import json
import os
from typing import Any, Dict, List, Optional

import click
import numpy as np
import torch
import tqdm
from PIL import Image

from cbc.caption import CAPTION_ENGINES_CLI
from cbc.caption.utils import postprocess_caption
from cbc.caption_by_committee import DEFAULT_CBC_PROMPT, get_prompt_for_candidates
from cbc.lm import LM_ENGINES_CLI, LM_LOCAL_ENGINES
from cbc.metrics import (
    compute_and_add_base_metrics,
    compute_and_add_clip_recall,
    compute_and_add_content_recall,
    compute_and_add_mauve_score,
)


@click.command()
@click.argument("dataset_json_path", type=click.Path(exists=True))
@click.option(
    "--caption-engine",
    type=click.Choice(CAPTION_ENGINES_CLI.keys()),  # type: ignore
    default="ofa",
    help="The underlying captioning model to use.",
)
@click.option(
    "--lm-engine",
    type=click.Choice(LM_ENGINES_CLI.keys()),  # type: ignore
    default="gpt3_davinci3",
    help="The LM to use.",
)
@click.option("--num-candidates", type=int, default=15, help="Number of candidates to generate for each image.")
@click.option("--candidate-temperature", type=float, default=1.0, help="Temperature to use when generating candidates.")
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_CBC_PROMPT,
    help="The prompt to use when generating candidates. Will load from a file if it exists.",
)
@click.option("--output-json-path", type=str, default="output.json", help="The path to save the output to.")
@click.option("--candidate-key", type=str, default="candidates", help="The key to use for the candidates.")
@click.option("--reference-key", type=str, default="references", help="The key to use for the references.")
@click.option("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
@click.option("--image-root-dir", type=str, default=None, help="The root directory for the images.")
@click.option("--overwrite-candidates", is_flag=True, help="Whether to overwrite the candidates if they already exist.")
def evaluate_dataset(
    dataset_json_path: str,
    caption_engine: str,
    lm_engine: str,
    num_candidates: int,
    candidate_temperature: float,
    prompt: str,
    output_json_path: str,
    candidate_key: str,
    reference_key: str,
    image_path_key: str,
    image_root_dir: Optional[str] = None,
    overwrite_candidates: bool = False,
) -> None:
    # TODO: Implement this

    # 1. Load the dataset (references + image paths)
    print(f"Loading dataset from {dataset_json_path}...")
    with open(dataset_json_path, "r") as f:
        samples: List[Dict[str, Any]] = json.load(f)

    # 1.1 Load the prompt (if not already loaded)
    if os.path.exists(prompt):
        print(f"Loading prompt from {prompt}...")
        with open(prompt, "r") as f:
            prompt = f.read().strip()

    # 2. Compute candidate captions for each image (If not already computed)
    print(f"Loading caption engine {caption_engine}...")
    captioner = CAPTION_ENGINES_CLI[caption_engine](
        device="cuda" if torch.cuda.is_available() else "cpu",
    )  # type: ignore
    print(f"Generating candidates using {caption_engine}...")
    for sample in tqdm.tqdm(samples):
        if sample.get(candidate_key, None) is None or overwrite_candidates:
            sample[candidate_key] = captioner(
                Image.open(os.path.join(image_root_dir or ".", sample[image_path_key])).convert("RGB"),
                n_captions=num_candidates,
                temperature=candidate_temperature,
            )
        # The baseline is always the first candidate
        if sample.get("baseline", None) is None or overwrite_candidates:
            sample["baseline"] = sample[candidate_key][0]  # type: ignore

    # 3. Compute the summary captions for each image (both candidate + reference summaries, if not already computed)
    print(f"Loading LM engine {lm_engine}...")
    if lm_engine in LM_LOCAL_ENGINES:
        lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        lm = LM_ENGINES_CLI[lm_engine]()

    print(f"Generating summaries using {lm_engine}...")
    for sample in tqdm.tqdm(samples):
        if sample.get("candidate_summary") is None or overwrite_candidates:
            sample["candidate_summary"] = postprocess_caption(
                lm.best(prompt=get_prompt_for_candidates(sample[candidate_key], prompt=prompt))
            )
        if sample.get("reference_summary") is None or overwrite_candidates:
            sample["reference_summary"] = postprocess_caption(
                lm.best(prompt=get_prompt_for_candidates(sample[reference_key], prompt=prompt))
            )

    # 4. Compute the metrics (Bleu, ROUGE, METEOR, CIDEr, SPICE) for each image (if not already computed)
    print("Computing base metrics...")
    samples = compute_and_add_base_metrics(samples, reference_key)

    # 5. Compute the overall Mauve score for each set of samples (if not already computed)
    print("Computing Mauve score...")
    samples = compute_and_add_mauve_score(samples, reference_key)

    # 6. Compute the CLIP Recall for each set of candidates (if not already computed)
    print("Computing CLIP recall...")
    samples = compute_and_add_clip_recall(samples, image_path_key, image_root_dir)

    # 7. Compute the Content Recall for each set of candidates (if not already computed)
    print("Computing Content recall...")
    samples = compute_and_add_content_recall(samples, reference_key)

    # 8. Aggregate the metrics across all images
    metrics = {
        "standard": {
            # Base Scores
            "candidate_summary_bleu_1": float(np.mean([s["scores"]["candidate_summary_bleu_1"] for s in samples])),
            "candidate_summary_bleu_2": float(np.mean([s["scores"]["candidate_summary_bleu_2"] for s in samples])),
            "candidate_summary_bleu_3": float(np.mean([s["scores"]["candidate_summary_bleu_3"] for s in samples])),
            "candidate_summary_bleu_4": float(np.mean([s["scores"]["candidate_summary_bleu_4"] for s in samples])),
            "candidate_summary_rouge": float(np.mean([s["scores"]["candidate_summary_rouge"] for s in samples])),
            "candidate_summary_cider": float(np.mean([s["scores"]["candidate_summary_cider"] for s in samples])),
            "reference_summary_bleu_1": float(np.mean([s["scores"]["reference_summary_bleu_1"] for s in samples])),
            "reference_summary_bleu_2": float(np.mean([s["scores"]["reference_summary_bleu_2"] for s in samples])),
            "reference_summary_bleu_3": float(np.mean([s["scores"]["reference_summary_bleu_3"] for s in samples])),
            "reference_summary_bleu_4": float(np.mean([s["scores"]["reference_summary_bleu_4"] for s in samples])),
            "reference_summary_rouge": float(np.mean([s["scores"]["reference_summary_rouge"] for s in samples])),
            "reference_summary_cider": float(np.mean([s["scores"]["reference_summary_cider"] for s in samples])),
            "baseline_bleu_1": float(np.mean([s["scores"]["baseline_bleu_1"] for s in samples])),
            "baseline_bleu_2": float(np.mean([s["scores"]["baseline_bleu_2"] for s in samples])),
            "baseline_bleu_3": float(np.mean([s["scores"]["baseline_bleu_3"] for s in samples])),
            "baseline_bleu_4": float(np.mean([s["scores"]["baseline_bleu_4"] for s in samples])),
            "baseline_rouge": float(np.mean([s["scores"]["baseline_rouge"] for s in samples])),
            "baseline_cider": float(np.mean([s["scores"]["baseline_cider"] for s in samples])),
            # Mauve Scores
            "candidate_summary_mauve": float(np.mean([s["scores"]["candidate_summary_mauve"] for s in samples])),
            "reference_summary_mauve": float(np.mean([s["scores"]["reference_summary_mauve"] for s in samples])),
            "baseline_mauve": float(np.mean([s["scores"]["baseline_mauve"] for s in samples])),
        },
        # CLIP Scores
        "clip_recall": {
            "candidate_summary_clip_recall_rank": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_rank"] for s in samples])
            ),
            "candidate_summary_clip_recall_mrr": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_mrr"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_1": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_1"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_5": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_5"] for s in samples])
            ),
            "candidate_summary_clip_recall_at_10": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_at_10"] for s in samples])
            ),
            "candidate_summary_clip_recall_max_rank": float(
                np.mean([s["scores"]["candidate_summary_clip_recall_max_rank"] for s in samples])
            ),
            "reference_summary_clip_recall_rank": float(
                np.mean([s["scores"]["reference_summary_clip_recall_rank"] for s in samples])
            ),
            "reference_summary_clip_recall_mrr": float(
                np.mean([s["scores"]["reference_summary_clip_recall_mrr"] for s in samples])
            ),
            "reference_summary_clip_recall_at_1": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_1"] for s in samples])
            ),
            "reference_summary_clip_recall_at_5": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_5"] for s in samples])
            ),
            "reference_summary_clip_recall_at_10": float(
                np.mean([s["scores"]["reference_summary_clip_recall_at_10"] for s in samples])
            ),
            "reference_summary_clip_recall_max_rank": float(
                np.mean([s["scores"]["reference_summary_clip_recall_max_rank"] for s in samples])
            ),
            "baseline_clip_recall_rank": float(np.mean([s["scores"]["baseline_clip_recall_rank"] for s in samples])),
            "baseline_clip_recall_mrr": float(np.mean([s["scores"]["baseline_clip_recall_mrr"] for s in samples])),
            "baseline_clip_recall_at_1": float(np.mean([s["scores"]["baseline_clip_recall_at_1"] for s in samples])),
            "baseline_clip_recall_at_5": float(np.mean([s["scores"]["baseline_clip_recall_at_5"] for s in samples])),
            "baseline_clip_recall_at_10": float(np.mean([s["scores"]["baseline_clip_recall_at_10"] for s in samples])),
            "baseline_clip_recall_max_rank": float(
                np.mean([s["scores"]["baseline_clip_recall_max_rank"] for s in samples])
            ),
        },
        # Content Scores
        "content_recall": {
            "candidate_summary_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_noun_recall"] for s in samples])
            ),
            "candidate_summary_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_verb_recall"] for s in samples])
            ),
            "candidate_summary_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_noun_fuzzy_recall"] for s in samples])
            ),
            "candidate_summary_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["candidate_summary_verb_fuzzy_recall"] for s in samples])
            ),
            "reference_summary_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_noun_recall"] for s in samples])
            ),
            "reference_summary_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_verb_recall"] for s in samples])
            ),
            "reference_summary_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_noun_fuzzy_recall"] for s in samples])
            ),
            "reference_summary_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["reference_summary_verb_fuzzy_recall"] for s in samples])
            ),
            "baseline_noun_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_noun_recall"] for s in samples])
            ),
            "baseline_verb_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_verb_recall"] for s in samples])
            ),
            "baseline_noun_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_noun_fuzzy_recall"] for s in samples])
            ),
            "baseline_verb_fuzzy_recall": float(
                np.mean([s["scores"]["content_recall"]["baseline_verb_fuzzy_recall"] for s in samples])
            ),
        },
    }

    # 8. Save the results to a JSON file
    with open(output_json_path, "w") as f:
        json.dump({"samples": samples, "metrics": metrics}, f, indent=2)

    # 9. Print the results to the console
    print(json.dumps(metrics, indent=2))
