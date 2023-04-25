import json
import os
from typing import Any, Dict, List, Optional

import click
import numpy as np
import torch
import tqdm
from PIL import Image
from vdtk.metrics.distribution import TriangleRankMetricScorer
from vdtk.metrics.distribution.distance import CIDERDDistance

from cbc.caption import CAPTION_ENGINES_CLI, CaptionEngine


def _compute_trm_cider(
    dataset: List[Dict[str, Any]],
    captioner: CaptionEngine,
    scorer: TriangleRankMetricScorer,
    temperature: float,
    num_candidates: int = 10,
    image_root_dir: Optional[str] = None,
    image_path_key: str = "image_path",
    candidate_key: str = "candidates",
    reference_key: str = "references",
    overwrite_candidates: bool = False,
) -> np.floating:
    candidates = {}
    references = {}
    for i, sample in enumerate(tqdm.tqdm(dataset)):
        if sample.get(candidate_key, None) is None or overwrite_candidates:
            candidates[i] = captioner(
                Image.open(os.path.join(image_root_dir or ".", sample[image_path_key])).convert("RGB"),
                n_captions=num_candidates,
                temperature=temperature,
            )
        else:
            candidates[i] = sample[candidate_key]
        references[i] = sample[reference_key]

    # Compute the TRM
    trm = scorer(candidates, references)

    # Return the mean of the test statistics
    return np.nanmean([t.test_statistic for t in trm.values() if t])


@click.command()
@click.argument("dataset_json_path", type=click.Path(exists=True))
@click.option(
    "--caption-engine",
    type=click.Choice(CAPTION_ENGINES_CLI.keys()),  # type: ignore
    default="ofa",
    help="The underlying captioning model to use.",
)
@click.option("--num-candidates", type=int, default=15, help="Number of candidates to generate for each image.")
@click.option("--candidate-key", type=str, default="candidates", help="The key to use for the candidates.")
@click.option("--reference-key", type=str, default="references", help="The key to use for the references.")
@click.option("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
@click.option("--image-root-dir", type=str, default=None, help="The root directory for the images.")
@click.option("--overwrite-candidates", is_flag=True, help="Whether to overwrite the candidates if they already exist.")
def main(
    dataset_json_path: str,
    caption_engine: str,
    num_candidates: int,
    candidate_key: str,
    reference_key: str,
    image_path_key: str,
    image_root_dir: Optional[str] = None,
    overwrite_candidates: bool = False,
):

    # Load the dataset
    with open(dataset_json_path) as f:
        dataset = json.load(f)

    # Load the captioner and scorer
    captioner = CAPTION_ENGINES_CLI[caption_engine](device="cuda" if torch.cuda.is_available() else "cpu")
    scorer = TriangleRankMetricScorer(
        distance_function=CIDERDDistance, num_null_samples=0, num_uk_samples=100, quiet=True, num_workers=24
    )

    trms = []
    temperatures = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    for temperature in temperatures:
        trm = _compute_trm_cider(
            dataset=dataset,
            captioner=captioner,
            scorer=scorer,
            temperature=temperature,
            num_candidates=num_candidates,
            image_root_dir=image_root_dir,
            image_path_key=image_path_key,
            candidate_key=candidate_key,
            reference_key=reference_key,
            overwrite_candidates=overwrite_candidates,
        )
        trms.append(trm)
        print(f"TRM (temperature={temperature}): {trm}")

    # Save the results as a numpy array
    np.save(f"trm_{caption_engine}.npy", np.array(trms))


if __name__ == "__main__":
    main()
