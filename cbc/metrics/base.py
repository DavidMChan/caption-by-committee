import itertools
from typing import Any, Dict, List

import mauve
import torch
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.cider.cider import Cider
from vdtk.metrics.rouge.rouge import Rouge
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer


def compute_and_add_base_metrics(samples: List[Dict[str, Any]], reference_key: str) -> List[Dict[str, Any]]:

    chyp = {i: [d["candidate_summary"]] for i, d in enumerate(samples)}
    rhyp = {i: [d["reference_summary"]] for i, d in enumerate(samples)}
    bhyp = {i: [d["baseline"]] for i, d in enumerate(samples)}
    cref = {i: d[reference_key] for i, d in enumerate(samples)}

    if samples:
        tokenizer = PTBTokenizer()

        # Tokenize the hypotheses and references
        chyp_tok = tokenizer.tokenize(chyp)
        rhyp_tok = tokenizer.tokenize(rhyp)
        bhyp_tok = tokenizer.tokenize(bhyp)
        cref_tok = tokenizer.tokenize(cref)

        # Compute the BLEU, ROUGE, and CIDEr scores
        c_bleu_score, c_b_full = Bleu().compute_score(cref_tok, chyp_tok)
        r_bleu_score, r_b_full = Bleu().compute_score(cref_tok, rhyp_tok)
        b_bleu_score, b_b_full = Bleu().compute_score(cref_tok, bhyp_tok)

        c_rouge_score, c_r_full = Rouge().compute_score(cref_tok, chyp_tok)
        r_rouge_score, r_r_full = Rouge().compute_score(cref_tok, rhyp_tok)
        b_rouge_score, b_r_full = Rouge().compute_score(cref_tok, bhyp_tok)

        c_cider_score, c_cd_full = Cider().compute_score(cref_tok, chyp_tok)
        r_cider_score, r_cd_full = Cider().compute_score(cref_tok, rhyp_tok)
        b_cider_score, b_cd_full = Cider().compute_score(cref_tok, bhyp_tok)

        # Update the scores
        for i, d in enumerate(samples):
            if d.get("scores") is None:
                d["scores"] = {}

            d["scores"]["candidate_summary_bleu_1"] = float(c_b_full[0][i])  # type: ignore
            d["scores"]["candidate_summary_bleu_2"] = float(c_b_full[1][i])  # type: ignore
            d["scores"]["candidate_summary_bleu_3"] = float(c_b_full[2][i])  # type: ignore
            d["scores"]["candidate_summary_bleu_4"] = float(c_b_full[3][i])  # type: ignore
            d["scores"]["candidate_summary_rouge"] = float(c_r_full[i])  # type: ignore
            d["scores"]["candidate_summary_cider"] = float(c_cd_full[i])  # type: ignore

            d["scores"]["reference_summary_bleu_1"] = float(r_b_full[0][i])  # type: ignore
            d["scores"]["reference_summary_bleu_2"] = float(r_b_full[1][i])  # type: ignore
            d["scores"]["reference_summary_bleu_3"] = float(r_b_full[2][i])  # type: ignore
            d["scores"]["reference_summary_bleu_4"] = float(r_b_full[3][i])  # type: ignore
            d["scores"]["reference_summary_rouge"] = float(r_r_full[i])  # type: ignore
            d["scores"]["reference_summary_cider"] = float(r_cd_full[i])  # type: ignore

            d["scores"]["baseline_bleu_1"] = float(b_b_full[0][i])  # type: ignore
            d["scores"]["baseline_bleu_2"] = float(b_b_full[1][i])  # type: ignore
            d["scores"]["baseline_bleu_3"] = float(b_b_full[2][i])  # type: ignore
            d["scores"]["baseline_bleu_4"] = float(b_b_full[3][i])  # type: ignore
            d["scores"]["baseline_rouge"] = float(b_r_full[i])  # type: ignore
            d["scores"]["baseline_cider"] = float(b_cd_full[i])  # type: ignore

    return samples


def compute_and_add_mauve_score(samples: List[Dict[str, Any]], reference_key: str) -> List[Dict[str, Any]]:

    chyp = {i: [d["candidate_summary"]] for i, d in enumerate(samples)}
    rhyp = {i: [d["reference_summary"]] for i, d in enumerate(samples)}
    bhyp = {i: [d["baseline"]] for i, d in enumerate(samples)}
    cref = {i: d[reference_key] for i, d in enumerate(samples)}

    all_candidates = list(itertools.chain.from_iterable(chyp.values()))
    all_references = list(itertools.chain.from_iterable(rhyp.values()))
    all_baselines = list(itertools.chain.from_iterable(bhyp.values()))
    all_crefs = list(itertools.chain.from_iterable(cref.values()))

    c_mauve = mauve.compute_mauve(
        p_text=all_candidates, q_text=all_crefs, device_id=0 if torch.cuda.is_available() else -1, verbose=False
    ).mauve
    r_mauve = mauve.compute_mauve(
        p_text=all_references, q_text=all_crefs, device_id=0 if torch.cuda.is_available() else -1, verbose=False
    ).mauve
    b_mauve = mauve.compute_mauve(
        p_text=all_baselines, q_text=all_crefs, device_id=0 if torch.cuda.is_available() else -1, verbose=False
    ).mauve

    for s in samples:
        if s.get("scores") is None:
            s["scores"] = {}

        s["scores"]["candidate_summary_mauve"] = float(c_mauve)
        s["scores"]["reference_summary_mauve"] = float(r_mauve)
        s["scores"]["baseline_mauve"] = float(b_mauve)

    return samples
