import json
import sys

with open(sys.argv[1], "r") as jf:
    data = json.load(jf)["metrics"]
    for k, v in data.items():
        for kk, vv in v.items():
            data[k][kk] = round(vv, 3)

print(
    f"{data['content_recall']['candidate_summary_noun_recall']} & {data['content_recall']['candidate_summary_verb_recall']} & {data['content_recall']['candidate_summary_noun_fuzzy_recall']} & {data['content_recall']['candidate_summary_verb_fuzzy_recall']} & {data['clip_recall']['candidate_summary_clip_recall_mrr']} & {data['clip_recall']['candidate_summary_clip_recall_at_1']} & {data['clip_recall']['candidate_summary_clip_recall_at_5']} & {data['clip_recall']['candidate_summary_clip_recall_at_10']}"
)

print(
    f"{data['content_recall']['reference_summary_noun_recall']} & {data['content_recall']['reference_summary_verb_recall']} & {data['content_recall']['reference_summary_noun_fuzzy_recall']} & {data['content_recall']['reference_summary_verb_fuzzy_recall']} & {data['clip_recall']['reference_summary_clip_recall_mrr']} & {data['clip_recall']['reference_summary_clip_recall_at_1']} & {data['clip_recall']['reference_summary_clip_recall_at_5']} & {data['clip_recall']['reference_summary_clip_recall_at_10']}"
)

print(
    f"{data['content_recall']['baseline_noun_recall']} & {data['content_recall']['baseline_verb_recall']} & {data['content_recall']['baseline_noun_fuzzy_recall']} & {data['content_recall']['baseline_verb_fuzzy_recall']} & {data['clip_recall']['baseline_clip_recall_mrr']} & {data['clip_recall']['baseline_clip_recall_at_1']} & {data['clip_recall']['baseline_clip_recall_at_5']} & {data['clip_recall']['baseline_clip_recall_at_10']}"
)
