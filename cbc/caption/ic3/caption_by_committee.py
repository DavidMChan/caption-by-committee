# flake8: noqa

from typing import List, Dict, Union, TypedDict, Optional

from PIL import Image
import logging
import torch
import numpy as np
from nltk.corpus import stopwords

from cbc.caption.base import CaptionEngine
from cbc.caption.utils import postprocess_caption
from cbc.lm.base import (
    LMEngine,
)
from cbc.lm.huggingface_local_engine import HuggingFaceLocalLMEngine
from cbc.lm.huggingface_llama_engine import HuggingFaceLlamaLMEngine
from cbc.plugins.base import ImagePlugin
from cbc.caption.ic3.guards import GUARDS
from cbc.utils.print_utils import print_tokens_weighted


DEFAULT_CBC_PROMPT = """This is a hard problem. Carefully summarize in ONE detailed sentence the following captions by different (possibly incorrect) people describing the same thing. Be sure to describe everything, and identify when you're not sure.{prompt_body}
Captions: {image_captions}.
{image_info}Summary:  I'm not sure, but the image is likely of"""


class LikelihoodOutput(TypedDict):
    log_probs: torch.Tensor
    ranks: torch.Tensor
    normalized_rank_score: torch.Tensor
    output_topk_tokens: List[List[str]]
    output_tokens: List[str]


FILTER_TOKENS = {
    "or",
    ",",
    "possibly",
    "probably",
    "likely",
    "maybe",
    "perhaps",
    "could",
    "might",
    "a",
    "and",
    "potentially",
}
FILTER_TOKENS.update(stopwords.words("english"))


def _sort_captions(input_captions: List[str], scores: List[LikelihoodOutput]) -> List[str]:

    # Generate the scores for each input caption
    output_scores = []
    for score in scores:
        # Filter out any tokens which are in the list of filter tokens
        # log_probs = score["log_probs"][1:]
        # Use rank instead of log_probs
        log_probs = [np.log2(60000 - r.cpu().item()) for r in score["ranks"][1:]]

        tokens = [t.replace("Ġ", "") for t in score["output_tokens"][1:]]
        scnf = list(zip(log_probs, tokens))
        sc = [(s, t) for s, t in zip(log_probs, tokens) if t not in FILTER_TOKENS]
        output_scores.append(np.mean(np.stack([s[0] for s in sc])))
        # Print the weighted tokens, and their weights
        # print_tokens_weighted(
        #     [f"{t} ({w:.3f})" for w, t in scnf], [s[0] for s in scnf], strike=[t in FILTER_TOKENS for t in tokens]
        # )
        print("Caption Score (Higher is better): {:.3f}".format(output_scores[-1]))
        print_tokens_weighted(
            [f"{t} ({w})" for w, t in zip(score["ranks"][1:], tokens)],
            [s[0] for s in scnf],
            strike=[t in FILTER_TOKENS for t in tokens],
        )

    # Sort input_captions by output_scores
    output_captions = [s for _, s in sorted(zip(output_scores, input_captions), reverse=True)]

    return output_captions


def get_prompt_for_candidates(
    candidates: List[str], prompt: str = DEFAULT_CBC_PROMPT, plugin_outputs: List[Dict[str, str]] = [], **kwargs: str
) -> str:
    """
    Generate a prompt for a list of candidates.
    """
    candidates = [postprocess_caption(c) for c in candidates]
    candidates_formatted = [f'"{c}"' for c in candidates]

    # Merge each of the plugin outputs
    plugin_output = {
        "prompt_body": "",
        "image_info": "",
    }
    for plugin_output_dict in plugin_outputs:
        for k, v in plugin_output_dict.items():
            if k not in plugin_output:
                plugin_output[k] = v
            else:
                plugin_output[k] += v

    # Make sure that the image_info plugin output ends with a newline if it exists
    if "image_info" in plugin_output and not plugin_output["image_info"].endswith("\n"):
        plugin_output["image_info"] += "\n"

    # Make sure that the prompt_body plugin output starts with a space if it exists, and doesn't end with whitespace
    if "prompt_body" in plugin_output:
        plugin_output["prompt_body"] = plugin_output["prompt_body"].strip()
        if not plugin_output["prompt_body"].startswith(" "):
            plugin_output["prompt_body"] = " " + plugin_output["prompt_body"]

    return prompt.format(image_captions=", ".join(candidates_formatted), **plugin_output, **kwargs)


def caption_by_committee(
    raw_image: Image.Image,
    caption_engine: CaptionEngine,
    lm_engine: LMEngine,
    caption_engine_temperature: float = 1.0,
    n_captions: int = 5,
    lm_prompt: str = DEFAULT_CBC_PROMPT,
    postprocess: str = "all",
    verbose: bool = False,
    plugins: List[ImagePlugin] = [],
    prompt_kwargs: Dict[str, str] = {},
    guard_failure_limit: int = 5,
    num_outputs: int = 1,
    lm_temperature: float = 1.0,
    select_best: bool = False,
    force_candidate_captions: Optional[List[str]] = None,
) -> Union[str, List[str]]:

    """
    Generate a caption for an image using a committee of captioning models.
    """
    if force_candidate_captions is not None:
        captions = force_candidate_captions
    else:
        captions = caption_engine(raw_image, n_captions=n_captions, temperature=caption_engine_temperature)

    logging.debug(f"Candidate Captions: {captions}")

    # TODO: This needs to be updated so that it's set on a per-model basis, and not based on classes.
    if postprocess == "default":
        if hasattr(lm_engine, "DEFAULT_POSTPROCESSOR"):
            postprocess = lm_engine.DEFAULT_POSTPROCESSOR
        elif isinstance(lm_engine, (HuggingFaceLocalLMEngine, HuggingFaceLlamaLMEngine)):
            postprocess = "all_truncate"
        else:
            postprocess = "all"

    # Get the plugin outputs
    plugin_outputs = [p(raw_image) for p in plugins]
    prompt = get_prompt_for_candidates(captions, prompt=lm_prompt, plugin_outputs=plugin_outputs, **prompt_kwargs)
    logging.debug(f"Prompt: {prompt}")

    if num_outputs == 1:
        summary = postprocess_caption(lm_engine.best(prompt), postprocess)

        # If the summary fails the guards, then try again
        guard_failure_count = 0
        guard_results = {k: guard(summary) for k, guard in GUARDS.items()}
        while any(guard_results.values()) and guard_failure_count < guard_failure_limit:
            logging.info(
                f'Caption "{summary}" failed one (or more) guards: {[k for k, v in guard_results.items() if v]}'
            )
            summary = postprocess_caption(
                lm_engine(prompt, n_completions=1, temperature=lm_temperature)[0], postprocess
            )
            guard_results = {k: guard(summary) for k, guard in GUARDS.items()}
            guard_failure_count += 1
        logging.debug(f"Final guard results: {guard_results}")

        return summary

    output_captions: List[str] = []
    failed_captions: List[str] = []
    for _ in range(guard_failure_limit):
        possible_summaries = lm_engine(
            prompt, n_completions=num_outputs - len(output_captions), temperature=lm_temperature
        )
        possible_summaries = [postprocess_caption(s, postprocess) for s in possible_summaries]
        for s in possible_summaries:
            guard_results = {k: guard(s) for k, guard in GUARDS.items()}
            if not any(guard_results.values()):
                output_captions.append(s)
            else:
                failed_captions.append(s)

        if len(output_captions) == num_outputs:
            break

    if len(output_captions) < num_outputs:
        logging.warning(
            f"Failed to generate {num_outputs} captions. Returning {len(output_captions)} captions instead. "
            f"Failed captions: {failed_captions}"
        )

    # Evaluate the output summaries using the likelihood under the caption engine, and sort them
    # in descending order of likelihood.add()
    evaluated_captions = [caption_engine.likelihood(raw_image, s) for s in output_captions]
    sorted_evaluated_captions = _sort_captions(output_captions, evaluated_captions)

    if select_best:
        return sorted_evaluated_captions[0]
    return sorted_evaluated_captions
