import json
import os
from typing import Any, Dict, List, Optional

import click
import torch
import tqdm
from PIL import Image

from cbc.caption import CAPTION_ENGINES_CLI
from cbc.caption.utils import postprocess_caption
from cbc.caption_by_comittee import DEFAULT_CBC_PROMPT, get_prompt_for_candidates
from cbc.lm import LM_ENGINES_CLI, LM_LOCAL_ENGINES


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
@click.option("--prompt", type=str, default=DEFAULT_CBC_PROMPT, help="The prompt to use when generating candidates.")
@click.option("--candidate-key", type=str, default="candidates", help="The key to use for the candidates.")
@click.option("--reference-key", type=str, default="references", help="The key to use for the references.")
@click.option("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
@click.option("--image-root-dir", type=str, default=None, help="The root directory for the images.")
def evaluate_dataset(
    dataset_json_path: str,
    caption_engine: str,
    lm_engine: str,
    num_candidates: int,
    candidate_temperature: float,
    prompt: str,
    candidate_key: str,
    reference_key: str,
    image_path_key: str,
    image_root_dir: Optional[str] = None,
) -> None:
    # TODO: Implement this

    # 1. Load the dataset (references + image paths)
    with open(dataset_json_path, "r") as f:
        samples: List[Dict[str, Any]] = json.load(f)

    # 2. Compute candidate captions for each image (If not already computed)
    captioner = CAPTION_ENGINES_CLI[caption_engine](
        device="cuda" if torch.cuda.is_available() else "cpu",
    )  # type: ignore
    for sample in tqdm.tqdm(samples):
        if sample[candidate_key] is None:
            sample[candidate_key] = captioner(
                Image.open(os.path.join(image_root_dir or ".", sample[image_path_key])).convert("RGB"),
                n_captions=num_candidates,
                temperature=candidate_temperature,
            )
        if sample["baseline"] is None:
            # The baseline is always the last candidate
            sample["baseline"] = sample[candidate_key][0]  # type: ignore

    # 3. Compute the summary captions for each image (both candidate + reference summaries, if not already computed)
    if lm_engine in LM_LOCAL_ENGINES:
        lm = LM_ENGINES_CLI[lm_engine](device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    else:
        lm = LM_ENGINES_CLI[lm_engine]()

    for sample in tqdm.tqdm(samples):
        if sample["candidate_summary"] is None:
            sample["candidate_summary"] = postprocess_caption(
                lm.best(prompt=get_prompt_for_candidates(sample[candidate_key], prompt=prompt))
            )
        if sample["reference_summary"] is None:
            sample["reference_summary"] = postprocess_caption(
                lm.best(prompt=get_prompt_for_candidates(sample[reference_key], prompt=prompt))
            )

    # 4. Compute the metrics (Bleu, ROUGE, METEOR, CIDEr, SPICE) for each image (if not already computed)

    # 5. Compute the overall Mauve score for each set of samples (if not already computed)

    # 6. Compute the CLIP Recall for each set of candidates (if not already computed)

    # 7. Compute the Content Recall for each set of candidates (if not already computed)

    # 8. Save the results to a JSON file

    # 9. Print the results to the console
