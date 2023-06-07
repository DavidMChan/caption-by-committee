from typing import Any, Dict, List
import tqdm
import enchant
import re

def has_listing_pattern(s):
    pattern_double = r'"[^"]+",\s*"[^"]+"(?:,\s*"[^"]+")*'
    pattern_single = r"[^']+',\s*'[^']+'(?:,\s*'[^']+')*"
    return bool(re.search(pattern_double, s)) or bool(re.search(pattern_single, s))

def compute_and_add_ocr_recall(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    english_dict = enchant.Dict('en_US')
    
    for sample in tqdm.tqdm(samples):
        ocr_list = sample['meta']['ocr_tokens']
        ocr_list = list(set([token.lower() for token in ocr_list]))
        # Only keep the OCR tokens that are in English dictionary
        ocr_list = [token for token in ocr_list if english_dict.check(token)]
        
        # Exclude those samples with no valid OCR tokens
        if len(ocr_list) == 0:
            sample['ocr_fraction'] = -1
            sample['ocr_mentioned'] = 0
            sample['gt_ocr_count'] = 0
            sample['has_listing'] = False
            sample['has_gt_ocr'] = False
            continue

        summary = sample['candidate_summary'].lower()
        
        # Count the number of ground truth OCR tokens that are mentioned in this summary
        sample_mentioned = 0
        for token in ocr_list:
            if token in summary:
                sample_mentioned += 1
        
        # Fraction of ground truth OCR tokens mentioned
        sample['ocr_fraction'] = sample_mentioned / len(ocr_list)
        # Number of ground truth OCR tokens mentioned
        sample['ocr_mentioned'] = sample_mentioned
        # Number of ground truth OCR tokens
        sample['gt_ocr_count'] = len(ocr_list)
        # Whether the summary has listing pattern
        sample['has_listing'] = has_listing_pattern(summary)
        # Whether the sample has at least one ground truth OCR tokens
        sample['has_gt_ocr'] = True
    
    return samples