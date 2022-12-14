from typing import Any, Dict, List, Tuple

import spacy
import tqdm

_NLP = None


def get_nlp() -> spacy.language.Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_lg")
        except OSError:
            _NLP = spacy.load("en_core_web_sm")

    return _NLP


def compute_object_roverlap(query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    nlp = get_nlp()

    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = set([token.text for token in query_doc if token.pos_ in POS])
    targets_objects = set([token.text for token in targets_doc if token.pos_ in POS])
    # Return the recall
    return float(len(set(query_objects).intersection(set(targets_objects))) / len(set(targets_objects)))


def compute_object_rdistance(query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    nlp = get_nlp()

    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = [token for token in query_doc if token.pos_ in POS]
    targets_objects = [token for token in targets_doc if token.pos_ in POS]

    query_uniq = set()
    targets_uniq = set()

    qos = []
    tos = []
    for token in query_objects:
        if token.text not in query_uniq:
            query_uniq.add(token.text)
            qos.append(token)
    for token in targets_objects:
        if token.text not in targets_uniq:
            targets_uniq.add(token.text)
            tos.append(token)

    metric = []
    for q in tos:
        sims = []
        for t in qos:
            sims.append(q.similarity(t))
        metric.append(max(sims) if sims else 0)

    return float(sum(metric) / (len(metric) + 1e-8))


def compute_and_add_content_recall(samples: List[Dict[str, Any]], reference_key: str) -> List[Dict[str, Any]]:

    for sample in tqdm.tqdm(samples):
        if "scores" not in sample:
            sample["scores"] = {}
        if "content_recall" not in sample["scores"]:
            sample["scores"]["content_recall"] = {}

        # Exact Recall
        sample["scores"]["content_recall"]["baseline_noun_recall"] = compute_object_roverlap(
            sample["baseline"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["baseline_verb_recall"] = compute_object_roverlap(
            sample["baseline"], sample[reference_key], POS=("VERB",)
        )
        sample["scores"]["content_recall"]["candidate_summary_noun_recall"] = compute_object_roverlap(
            sample["candidate_summary"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["candidate_summary_verb_recall"] = compute_object_roverlap(
            sample["candidate_summary"], sample[reference_key], POS=("VERB",)
        )
        sample["scores"]["content_recall"]["reference_summary_noun_recall"] = compute_object_roverlap(
            sample["reference_summary"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["reference_summary_verb_recall"] = compute_object_roverlap(
            sample["reference_summary"], sample[reference_key], POS=("VERB",)
        )

        # Fuzzy Recall
        sample["scores"]["content_recall"]["baseline_noun_fuzzy_recall"] = compute_object_rdistance(
            sample["baseline"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["baseline_verb_fuzzy_recall"] = compute_object_rdistance(
            sample["baseline"], sample[reference_key], POS=("VERB",)
        )
        sample["scores"]["content_recall"]["candidate_summary_noun_fuzzy_recall"] = compute_object_rdistance(
            sample["candidate_summary"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["candidate_summary_verb_fuzzy_recall"] = compute_object_rdistance(
            sample["candidate_summary"], sample[reference_key], POS=("VERB",)
        )
        sample["scores"]["content_recall"]["reference_summary_noun_fuzzy_recall"] = compute_object_rdistance(
            sample["reference_summary"], sample[reference_key], POS=("NOUN", "PROPN")
        )
        sample["scores"]["content_recall"]["reference_summary_verb_fuzzy_recall"] = compute_object_rdistance(
            sample["reference_summary"], sample[reference_key], POS=("VERB",)
        )

    return samples
