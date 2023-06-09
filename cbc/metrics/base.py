import itertools
from typing import Any, Dict, List

import mauve
import torch
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.cider.cider import Cider
from vdtk.metrics.rouge.rouge import Rouge
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer


def _cm_kys(samples: List[Dict[str, Any]], c_key: str, ref_key: str) -> List[Dict[str, Any]]:
    chyp = {i: [d[c_key]] for i, d in enumerate(samples)}
    cref = {i: d[ref_key] for i, d in enumerate(samples)}

    tokenizer = PTBTokenizer()

    # Tokenize the hypotheses and references
    chyp_tok = tokenizer.tokenize(chyp)
    cref_tok = tokenizer.tokenize(cref)

    # Compute the BLEU, ROUGE, and CIDEr scores
    c_bleu_score, c_b_full = Bleu().compute_score(cref_tok, chyp_tok)
    c_rouge_score, c_r_full = Rouge().compute_score(cref_tok, chyp_tok)
    c_cider_score, c_c_full = Cider().compute_score(cref_tok, chyp_tok)

    # Update the scores
    for i, d in enumerate(samples):
        if d.get("scores") is None:
            d["scores"] = {}

        d["scores"][f"{c_key}_bleu_1"] = float(c_b_full[0][i])
        d["scores"][f"{c_key}_bleu_2"] = float(c_b_full[1][i])
        d["scores"][f"{c_key}_bleu_3"] = float(c_b_full[2][i])
        d["scores"][f"{c_key}_bleu_4"] = float(c_b_full[3][i])
        d["scores"][f"{c_key}_rouge"] = float(c_r_full[i])
        d["scores"][f"{c_key}_cider"] = float(c_c_full[i])

    return samples


def _cm_kys_mv(samples: List[Dict[str, Any]], c_key: str, ref_key: str) -> List[Dict[str, Any]]:
    chyp = {i: [d[c_key]] for i, d in enumerate(samples)}
    cref = {i: d[ref_key] for i, d in enumerate(samples)}

    all_candidates = list(itertools.chain.from_iterable(chyp.values()))
    all_crefs = list(itertools.chain.from_iterable(cref.values()))

    c_mauve = mauve.compute_mauve(
        p_text=all_candidates, q_text=all_crefs, device_id=0 if torch.cuda.is_available() else -1, verbose=False
    ).mauve

    for s in samples:
        if s.get("scores") is None:
            s["scores"] = {}
        s["scores"][f"{c_key}_mauve"] = float(c_mauve)

    return samples


def compute_and_add_base_metrics(samples: List[Dict[str, Any]], reference_key: str) -> List[Dict[str, Any]]:
    # Compute the BLEU, ROUGE, and CIDEr scores
    if "candidate_summary" in samples[0]:
        samples = _cm_kys(samples, "candidate_summary", reference_key)
    if "reference_summary" in samples[0]:
        samples = _cm_kys(samples, "reference_summary", reference_key)
    if "baseline" in samples[0]:
        samples = _cm_kys(samples, "baseline", reference_key)

    return samples


def compute_and_add_mauve_score(samples: List[Dict[str, Any]], reference_key: str) -> List[Dict[str, Any]]:
    if "candidate_summary" in samples[0]:
        samples = _cm_kys_mv(samples, "candidate_summary", reference_key)
    if "reference_summary" in samples[0]:
        samples = _cm_kys_mv(samples, "reference_summary", reference_key)
    if "baseline" in samples[0]:
        samples = _cm_kys_mv(samples, "baseline", reference_key)

    return samples
