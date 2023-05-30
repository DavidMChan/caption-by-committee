from typing import Dict, List


def get_openai_hallucination_prompt_multiple_captions(target_captions: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are an assistant that parses objects from a set of captions. Given a set of captions, you list ALL the objects present in the image or photo described by the captions. Do not include an object's attributes or adjectives. Do not repeat objects.",
        },
        {
            "role": "user",
            "content": """
Captions:
- A baseball player runs into home plate during a game.
- A man is running the bases in a baseball game.
- A group of players playing a baseball game.
- A man is running to home base in a baseball game.
- A man in a white shirt and gray pants walks toward a grassy area as kids in baseball uniforms and an umpire are near him.

Objects:""",
        },
        {
            "role": "assistant",
            "content": """
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
- umpire""",
        },
        {
            "role": "user",
            "content": """
Captions:
- Several people riding on a motorcycle with an umbrella open.
- Couples riding motor cycles carrying umbrellas and people sitting at tables.
- A group of people riding scooters while holding umbrellas.
- Some tables and umbrellas sitting next to a building.
- Pedestrians and motorcyclists near an open outdoor market.

Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- people
- motorcycle
- umbrella
- table
- scooter
- building
- market
- pedestrian
- motorcyclist""",
        },
        {
            "role": "user",
            "content": """
Captions:
- Two black plastic containers filled with meat, gravy and veggies.
- Two small containers on a table, one with veggies and one with a main course.
- A lunch box with raw vegetables, toys and an entree on a terry towel.
- A bunch of veggies are in a small black bowl.
- Vegetables and other types of food are in two black containers.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
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
- food""",
        },
        {
            "role": "user",
            "content": """
Captions:
- A person standing next to the water and a umbrella.
- The man rides the skateboard next to the water and the umbrella.
- A man standing on a pier overlooking the lake with an umbrella nearby.
- A man riding a skate board on railings near waterfront.
- A person is standing at the side of a big lake.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- water
- umbrella
- man
- skateboard
- pier
- lake
- railing
- waterfront
- person""",
        },
        {
            "role": "user",
            "content": f"""
Captions:
{target_captions}
Objects:""",
        },
    ]


def get_openai_hallucination_prompt_single_caption(target_caption: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are an assistant that parses objects from a caption. Given a caption, you list ALL the objects present in the image or photo described by the caption. Do not include an object's attributes or adjectives. Do not repeat objects.",
        },
        {
            "role": "user",
            "content": """
Caption: An assortment of fruits and vegetables, including lime, pickles, carrots, and radishes, arranged in a bowl or on a piece of aluminum foil.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- fruit
- vegetable
- lime
- pickle
- carrot
- radish
- bowl or foil
""",
        },
        {
            "role": "user",
            "content": """
Caption: A desk with two computer monitors, a laptop, a gray keyboard, printer, and possibly other items, such as papers, notebooks, and other clutter.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- desk
- monitor
- laptop
- keyboard
- printer
- paper (possibly)
- notebook (possibly)
""",
        },
        {
            "role": "user",
            "content": """
Caption: A white table with a globe, lamp, pencil container, a bunch of bananas, and possibly a large violin case on it.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- table
- globe
- lamp
- container
- banana
- case (possibly)
""",
        },
        {
            "role": "user",
            "content": """
Caption: Two women in white shirts playing a game of tennis on a court, with one of them wiping sweat from her face, with a pained facial expression, possibly due to frustration from losing a match, with people spectating.
Objects:""",
        },
        {
            "role": "assistant",
            "content": """
- woman
- shirt
- game
- court
- face
- people
""",
        },
        {
            "role": "user",
            "content": f"""
Caption: {target_caption}
Objects:""",
        },
    ]
