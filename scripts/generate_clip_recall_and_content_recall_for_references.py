import json

import click
import numpy as np
import tqdm

from cbc.metrics.clip_score import _compute_rank, _get_image_feature_db
from cbc.metrics.content_score import exact_overlap, fuzzy_overlap


def compute_clip_recall(dataset, image_path_key, image_root, reference_key) -> None:

    print("Creating feature database...")
    feature_db = _get_image_feature_db(dataset, image_path_key, image_root)

    print("Computing CLIP recall...")
    for index, sample in enumerate(tqdm.tqdm(dataset)):
        sample_mrr = []
        sample_recall_at_1 = []
        sample_recall_at_5 = []
        sample_recall_at_10 = []
        for ref in sample[reference_key]:
            reference_ranks = _compute_rank(index, feature_db, ref)
            sample_mrr.append(np.mean(1 / reference_ranks))
            sample_recall_at_1.append(np.mean(reference_ranks <= 1))
            sample_recall_at_5.append(np.mean(reference_ranks <= 5))
            sample_recall_at_10.append(np.mean(reference_ranks <= 10))
        sample["mrr"] = np.mean(sample_mrr)
        sample["recall_at_1"] = np.mean(sample_recall_at_1)
        sample["recall_at_5"] = np.mean(sample_recall_at_5)
        sample["recall_at_10"] = np.mean(sample_recall_at_10)

    # Print a summary of the results
    print(f"Mean MRR: {np.mean([s['mrr'] for s in dataset])}")
    print(f"Mean Recall@1: {np.mean([s['recall_at_1'] for s in dataset])}")
    print(f"Mean Recall@5: {np.mean([s['recall_at_5'] for s in dataset])}")
    print(f"Mean Recall@10: {np.mean([s['recall_at_10'] for s in dataset])}")


def compute_content_recall(dataset, reference_key):

    for sample in tqdm.tqdm(dataset):
        sample_n_overlap = []
        sample_n_fuzzy_overlap = []
        sample_v_overlap = []
        sample_v_fuzzy_overlap = []
        for ref in sample[reference_key]:
            other_refs = [r for r in sample[reference_key] if r != ref]
            sample_n_overlap.append(exact_overlap(ref, other_refs, POS=("NOUN", "PROPN")))
            sample_n_fuzzy_overlap.append(fuzzy_overlap(ref, other_refs, POS=("NOUN", "PROPN")))
            sample_v_overlap.append(exact_overlap(ref, other_refs, POS=("VERB",)))
            sample_v_fuzzy_overlap.append(fuzzy_overlap(ref, other_refs, POS=("VERB",)))
        sample["n_overlap"] = np.mean(sample_n_overlap)
        sample["n_fuzzy_overlap"] = np.mean(sample_n_fuzzy_overlap)
        sample["v_overlap"] = np.mean(sample_v_overlap)
        sample["v_fuzzy_overlap"] = np.mean(sample_v_fuzzy_overlap)

    # Print a summary of the results
    print(f"Mean Noun Exact Overlap: {np.mean([s['n_overlap'] for s in dataset])}")
    print(f"Mean Noun Fuzzy Overlap: {np.mean([s['n_fuzzy_overlap'] for s in dataset])}")
    print(f"Mean Verb Exact Overlap: {np.mean([s['v_overlap'] for s in dataset])}")
    print(f"Mean Verb Fuzzy Overlap: {np.mean([s['v_fuzzy_overlap'] for s in dataset])}")


@click.command()
@click.argument("dataset-json-path", type=click.Path(exists=True))
@click.option("--image-path-key", type=str, default="image_path")
@click.option("--image-root", type=click.Path(exists=True), default="./")
@click.option("--reference-key", type=str, default="references")
def main(
    dataset_json_path: str,
    image_path_key: str = "image_path",
    image_root: str = "./",
    reference_key: str = "references",
) -> None:

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)
        if isinstance(dataset, dict):
            dataset = dataset["samples"]

    # Compute the CLIP recall
    compute_clip_recall(dataset, image_path_key, image_root, reference_key)

    # Compute the content recall
    compute_content_recall(dataset, reference_key)


if __name__ == "__main__":
    main()
