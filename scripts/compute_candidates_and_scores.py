import json
import os
from typing import Optional

import click
import torch
import tqdm
from PIL import Image

from cbc.caption import CAPTION_ENGINES_CLI
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
@click.option("--num-candidates", type=int, default=15, help="Number of candidates to generate for each image.")
@click.option("--candidate-temperature", type=float, default=1.0, help="Temperature to use when generating candidates.")
@click.option("--output-json-path", type=str, default="output.json", help="The path to save the output to.")
@click.option("--candidate-key", type=str, default="candidates", help="The key to use for the candidates.")
@click.option("--reference-key", type=str, default="references", help="The key to use for the references.")
@click.option("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
@click.option("--image-root-dir", type=str, default=None, help="The root directory for the images.")
@click.option("--overwrite-candidates", is_flag=True, help="Whether to overwrite the candidates if they already exist.")
def main(
    dataset_json_path: str,
    caption_engine: str,
    num_candidates: int,
    candidate_temperature: float,
    output_json_path: str,
    candidate_key: str,
    reference_key: str,
    image_path_key: str,
    image_root_dir: Optional[str] = None,
    overwrite_candidates: bool = False,
) -> None:
    # 1. Load the Karpathy split of the MSCOCO dataset
    with open(dataset_json_path) as f:
        samples = json.load(f)

    # 2. Generate the captions using the specified candidate caption engine
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

    # Compute the scores for the dataset
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

    # Write the dataset to a JSON file
    with open(output_json_path, "w") as f:
        json.dump(samples, f)


if __name__ == "__main__":
    main()
