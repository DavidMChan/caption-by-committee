from typing import Any, Dict, List

import numpy as np
import tqdm
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer


def self_bleu(candidates: List[str]) -> float:
    tokenizer = PTBTokenizer()

    if len(candidates) <= 1:
        return 1.0

    # Tokenize the candidates
    chyp = {}
    cref = {}
    for i in range(len(candidates)):
        # Hypothesis is the index i, reference is all indices except i
        chyp[i] = [candidates[i]]
        cref[i] = [candidates[j] for j in range(len(candidates)) if j != i]

    # Compute the BLEU scores
    chyp_tok = tokenizer.tokenize(chyp)
    cref_tok = tokenizer.tokenize(cref)
    _, c_b_full = Bleu().compute_score(cref_tok, chyp_tok)
    return float(np.mean(list(c_b_full[0].values())))


def compute_and_add_self_bleu(
    samples: List[Dict[str, Any]], candidate_key: str, reference_key: str
) -> List[Dict[str, Any]]:

    for sample in tqdm.tqdm(samples):
        if "scores" not in sample:
            sample["scores"] = {}
        if "self_bleu" not in sample["scores"]:
            sample["scores"]["self_bleu"] = {}
            sample["scores"]["self_bleu"]["candidates"] = self_bleu(sample[candidate_key])
            sample["scores"]["self_bleu"]["references"] = self_bleu(sample[reference_key])

    return samples
