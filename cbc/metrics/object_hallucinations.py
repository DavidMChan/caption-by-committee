from typing import Any, Dict, List

import numpy as np
import openai
import spacy
import tqdm
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from cbc.lm import ChatGPT, OpenAI


def extract_root_noun(text, nlp):
    doc = nlp(text)

    root_noun = next((token for token in doc if token.head == token), None)
    # Return the lemma of the root noun
    return root_noun.lemma_ if root_noun is not None else None

def parse_objects(obj_str, model, nlp):
    if obj_str is None or obj_str == "":
        return [], []

    res = []
    or_indices = []

    for obj in obj_str.split("\n"):
        if len(obj) < 3:
           continue 

        if obj[0] != '-':
            continue
        if 'possibly' in obj:
            continue

        obj = obj[1:].strip()

        if '(' in obj:
            obj = obj[:obj.index('(')].strip()

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

def extract_objects_single_caption(target_caption):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that parses objects from a caption. Given a caption, you list ALL the objects present in the image or photo described by the caption. Do not include an object's attributes or adjectives. Do not repeat objects."},
            {"role": "user", "content": """
Caption: An assortment of fruits and vegetables, including lime, pickles, carrots, and radishes, arranged in a bowl or on a piece of aluminum foil.
Objects:"""},
            {"role": "assistant", "content": """
- fruit
- vegetable
- lime
- pickle
- carrot
- radish
- bowl or foil
            """},
            {"role": "user", "content": """
Caption: A desk with two computer monitors, a laptop, a gray keyboard, printer, and possibly other items, such as papers, notebooks, and other clutter.
Objects:"""},
            {"role": "assistant", "content": """
- desk
- monitor
- laptop
- keyboard
- printer
- paper (possibly)
- notebook (possibly)
            """},
            {"role": "user", "content": """
Caption: A white table with a globe, lamp, pencil container, a bunch of bananas, and possibly a large violin case on it.
Objects:"""},
            {"role": "assistant", "content": """
- table
- globe
- lamp
- container
- banana
- case (possibly)
            """},
            {"role": "user", "content": """
Caption: Two women in white shirts playing a game of tennis on a court, with one of them wiping sweat from her face, with a pained facial expression, possibly due to frustration from losing a match, with people spectating.
Objects:"""},
            {"role": "assistant", "content": """
- woman
- shirt
- game
- court
- face
- people
            """},
            {"role": "user", "content": f"""
Caption: {target_caption}
Objects:"""},
        ]
    )
    
    OpenAI.USAGE += int(completion.usage.total_tokens) * ChatGPT.COST_PER_TOKEN

    return completion.choices[0].message.content

def extract_objects_multiple_captions(references):
    target_caption = "\n".join([f"- {r}" for r in references])
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that parses objects from a set of captions. Given a set of captions, you list ALL the objects present in the image or photo described by the captions. Do not include an object's attributes or adjectives. Do not repeat objects."},
            {"role": "user", "content": """
Captions:
- A baseball player runs into home plate during a game.
- A man is running the bases in a baseball game. 
- A group of players playing a baseball game.
- A man is running to home base in a baseball game.
- A man in a white shirt and gray pants walks toward a grassy area as kids in baseball uniforms and an umpire are near him.

Objects:"""},
            {"role": "assistant", "content": """
- player
- plate
- game
- man
- base
- group
- shirt
- pant
- grass
- kid
- uniform
- umpire"""},
            {"role": "user", "content": """
Captions:
- Several people riding on a motorcycle with an umbrella open.
- Couples riding motor cycles carrying umbrellas and people sitting at tables.
- A group of people riding scooters while holding umbrellas.
- Some tables and umbrellas sitting next to a building.
- Pedestrians and motorcyclists near an open outdoor market.

Objects:"""},
            {"role": "assistant", "content": """
- people
- motorcycle
- umbrella
- table
- scooter
- building
- market
- pedestrian
- motorcyclist"""},
            {"role": "user", "content": """
Captions:
- Two black plastic containers filled with meat, gravy and veggies.
- Two small containers on a table, one with veggies and one with a main course.
- A lunch box with raw vegetables, toys and an entree on a terry towel.
- A bunch of veggies are in a small black bowl.
- Vegetables and other types of food are in two black containers.
Objects:"""},
            {"role": "assistant", "content": """
- container
- meat
- gravy
- vegetable
- lunch
- box
- toy
- entree
- towel
- bowl
- food"""},
            {"role": "user", "content": """
Captions:
- A person standing next to the water and a umbrella.
- The man rides the skateboard next to the water and the umbrella.
- A man standing on a pier overlooking the lake with an umbrella nearby.
- A man riding a skate board on railings near waterfront. 
- A person is standing at the side of a big lake.
Objects:"""},
            {"role": "assistant", "content": """
- water
- umbrella
- man
- skateboard
- pier
- lake
- railing
- waterfront
- person"""},
            {"role": "user", "content": f"""
Captions: 
{target_caption}
Objects:"""},
        ]
    )

    OpenAI.USAGE += int(completion.usage.total_tokens) * ChatGPT.COST_PER_TOKEN

    return completion.choices[0].message.content

def compute_and_add_object_hallucinations(samples: List[Dict[str, Any]], candidate_key: str, reference_key: str, semantic_similarity_threshold: float = .65
) -> List[Dict[str, Any]]:
    object_count = 0
    hallucinated_object_count = 0

    sample_count = 0
    hallucinated_sample_count = 0

    hungarian_matching_scores = []

    candidate_objects_key = f"{candidate_key}_objects"
    reference_objects_key = f"{reference_key}_objects"

    model = SentenceTransformer('all-mpnet-base-v2')
    text_embedding = model.encode
    nlp = spacy.load('en_core_web_lg')

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
            idx2 = obj_or_indices[j+1]

            target_obj1 = target_objects[idx1]
            target_obj2 = target_objects[idx2]

            if max_similarity[idx1] < semantic_similarity_threshold and max_similarity[idx2] < semantic_similarity_threshold and target_obj1 not in references and target_obj2 not in references:
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