from typing import Any, Dict, List

import numpy as np
import openai
import spacy
import tqdm
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from cbc.lm import ChatGPT, OpenAI
from cbc.metrics.hallucination.prompt import (
    get_openai_hallucination_prompt_multiple_captions,
    get_openai_hallucination_prompt_single_caption,
)


def extract_root_noun(text, nlp):
    doc = nlp(text)
    root_noun = next((token for token in doc if token.head == token), None)
    # Return the lemma of the root noun
    return root_noun.lemma_ if root_noun is not None else None


def parse_objects(obj_str, model, nlp):
    """Parse a string of objects and extract their root nouns using a spaCy NLP model.

    Args:
        obj_str (str): A string containing objects separated by newlines. Objects may be single nouns or two nouns separated by "or".
        model (str): The name of the spaCy NLP model to use.
        nlp (spacy.lang.<lang>.<Lang>): The spaCy NLP object to use for extracting root nouns.

    Returns:
        A tuple containing:
        - A list of the root nouns extracted from the input string, in lowercase.
        - A list of indices into the first list that correspond to the start of "or" clauses in the input string.

    Examples:
        >>> nlp = spacy.load("en_core_web_sm")
        >>> parse_objects("apple\norange\nbanana", "en_core_web_sm", nlp)
        ['apple', 'orange', 'banana'], []

        >>> parse_objects("- cat\n- dog\n- parrot or macaw", "en_core_web_sm", nlp)
        ['cat', 'dog', 'parrot', 'macaw'], [2]
    """
    if obj_str is None or obj_str == "":
        return [], []

    res = []
    or_indices = []

    for obj in obj_str.split("\n"):
        # Continue if the line is too short, doesn't start with a dash, or contains "possibly"
        if len(obj) < 3 or obj[0] != "-" or "possibly" in obj:
            continue

        # Parse the object from the string
        obj = obj[1:].strip()

        if "(" in obj:
            obj = obj[: obj.index("(")].strip()

        if " or " in obj:
            if len(obj.split(" or ")) != 2:
                continue
            obj1, obj2 = obj.split(" or ")

            obj1 = extract_root_noun(obj1, nlp)
            obj2 = extract_root_noun(obj2, nlp)

            if obj1 in res or obj2 in res:
                continue

            if obj1 is not None and obj2 is not None:
                or_indices += [len(res), len(res) + 1]
                obj1 = obj1.lower()
                obj2 = obj2.lower()
                res.extend((obj1, obj2))
        else:
            obj = extract_root_noun(obj, nlp)
            obj = obj.lower()
            if obj is not None and obj not in res:
                res.append(obj)

    return res, or_indices


def extract_objects_single_caption(target_caption: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=get_openai_hallucination_prompt_single_caption(target_caption),
    )
    OpenAI.USAGE += int(completion.usage.total_tokens) * ChatGPT.COST_PER_TOKEN
    return completion.choices[0].message.content


def extract_objects_multiple_captions(references: List[str]) -> str:
    target_caption = "\n".join([f"- {r}" for r in references])
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=get_openai_hallucination_prompt_multiple_captions(target_caption),
    )
    OpenAI.USAGE += int(completion.usage.total_tokens) * ChatGPT.COST_PER_TOKEN
    return completion.choices[0].message.content


def compute_and_add_object_hallucinations(
    samples: List[Dict[str, Any]], candidate_key: str, reference_key: str, semantic_similarity_threshold: float = 0.65
) -> List[Dict[str, Any]]:
    object_count = 0
    hallucinated_object_count = 0

    sample_count = 0
    hallucinated_sample_count = 0

    hungarian_matching_scores = []

    candidate_objects_key = f"{candidate_key}_objects"
    reference_objects_key = f"{reference_key}_objects"

    model = SentenceTransformer("all-mpnet-base-v2")
    text_embedding = model.encode
    nlp = spacy.load("en_core_web_lg")

    for i in tqdm.tqdm(range(len(samples)), desc="Computing"):
        sample = samples[i]
        if "hallucinated_object_rate" in sample:
            continue

        target = sample[candidate_key]

        # If objects have not been extracted, extract them
        if candidate_objects_key not in sample:
            sample[candidate_objects_key] = extract_objects_single_caption(target)
            samples[i][candidate_objects_key] = sample[candidate_objects_key]
        target_objects = sample[candidate_objects_key]

        if reference_objects_key not in sample:
            if type(sample[reference_key]) == list:
                sample[reference_objects_key] = extract_objects_multiple_captions(sample[reference_key])
            else:
                sample[reference_objects_key] = extract_objects_single_caption(sample[reference_key])
            samples[i][reference_objects_key] = sample[reference_objects_key]
        reference_objects = sample[reference_objects_key]

        if type(sample[reference_key]) == list:
            references = "\n".join(sample[reference_key]).lower()
        else:
            references = sample[reference_key].lower()

        hallucinated_objects = []

        reference_objects, _ = parse_objects(reference_objects, model, nlp)
        target_objects, obj_or_indices = parse_objects(target_objects, model, nlp)

        if len(reference_objects) == 0 or len(target_objects) == 0:
            continue

        reference_encodings = text_embedding(reference_objects)
        target_encodings = text_embedding(target_objects)

        similarity = target_encodings @ reference_encodings.T
        row_ind, col_ind = linear_sum_assignment(similarity, maximize=True)

        hm_score = np.mean(similarity[row_ind, col_ind])
        hungarian_matching_scores.append(hm_score)

        max_similarity = np.max(similarity, axis=1)

        hallucinated_objects = []
        for j in range(len(target_objects)):
            if j in obj_or_indices:
                continue
            if max_similarity[j] < semantic_similarity_threshold and target_objects[j] not in references:
                hallucinated_objects.append(target_objects[j])

        for j in range(0, len(obj_or_indices), 2):
            idx1 = obj_or_indices[j]
            idx2 = obj_or_indices[j + 1]

            target_obj1 = target_objects[idx1]
            target_obj2 = target_objects[idx2]

            if (
                max_similarity[idx1] < semantic_similarity_threshold
                and max_similarity[idx2] < semantic_similarity_threshold
                and target_obj1 not in references
                and target_obj2 not in references
            ):
                hallucinated_objects.append(f"{target_obj1} or {target_obj2}")

        hallucinated_object_count += len(hallucinated_objects)
        object_count += len(target_objects)

        samples[i]["hallucinated_objects"] = hallucinated_objects
        samples[i]["hallucinated_object_count"] = len(hallucinated_objects)
        samples[i]["scores"]["hallucinated_object_rate"] = len(hallucinated_objects) / len(target_objects)

        samples[i]["scores"]["hungarian_matching_score"] = float(hm_score)
        samples[i]["object_count"] = len(target_objects)

        sample_count += 1
        if hallucinated_objects:
            hallucinated_sample_count += 1

    float(np.mean(hungarian_matching_scores))
    float(hallucinated_object_count / object_count)
    float(hallucinated_sample_count / sample_count)
    float(hallucinated_object_count / hallucinated_sample_count)

    return samples
