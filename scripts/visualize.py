import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT_PATH = "/home/davidchan/Projects/cbc/scripts/output.json"

with open(ROOT_PATH, "r") as jf:
    data = json.load(jf)
    generated_captions = data["samples"]


# Display the data
st.header(f"Visualizing Results: {ROOT_PATH}")


def _stm(display, metric, cd, caps):
    mean_value = np.mean([i["scores"][f"{cd}{metric}"] for i in caps])
    diff = mean_value - np.mean([i["scores"][f"baseline{metric}"] for i in caps])
    return st.metric(display, f"{mean_value:.3f}", f"{diff:+.3f}")


with st.expander("Metrics"):
    st.subheader("Standard Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Candidate Summary")
        _stm("BLEU@1", "_bleu_1", "candidate_summary", generated_captions)
        _stm("BLEU@2", "_bleu_2", "candidate_summary", generated_captions)
        _stm("BLEU@3", "_bleu_3", "candidate_summary", generated_captions)
        _stm("BLEU@4", "_bleu_4", "candidate_summary", generated_captions)
        _stm("ROUGE", "_rouge", "candidate_summary", generated_captions)
        _stm("CIDEr", "_cider", "candidate_summary", generated_captions)
        _stm("Mauve", "_mauve", "candidate_summary", generated_captions)
    with c2:
        st.write("Reference Summary")
        _stm("BLEU@1", "_bleu_1", "reference_summary", generated_captions)
        _stm("BLEU@2", "_bleu_2", "reference_summary", generated_captions)
        _stm("BLEU@3", "_bleu_3", "reference_summary", generated_captions)
        _stm("BLEU@4", "_bleu_4", "reference_summary", generated_captions)
        _stm("ROUGE", "_rouge", "reference_summary", generated_captions)
        _stm("CIDEr", "_cider", "reference_summary", generated_captions)
        _stm("Mauve", "_mauve", "reference_summary", generated_captions)
    with c3:
        st.write("Baseline")
        _stm("BLEU@1", "_bleu_1", "baseline", generated_captions)
        _stm("BLEU@2", "_bleu_2", "baseline", generated_captions)
        _stm("BLEU@3", "_bleu_3", "baseline", generated_captions)
        _stm("BLEU@4", "_bleu_4", "baseline", generated_captions)
        _stm("ROUGE", "_rouge", "baseline", generated_captions)
        _stm("CIDEr", "_cider", "baseline", generated_captions)
        _stm("Mauve", "_mauve", "baseline", generated_captions)


with st.expander("CLIP Recall Metrics"):
    st.subheader("CLIP Recall")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Candidate Summary Rank",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_rank']:.3f}",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_rank'] - data['metrics']['clip_recall']['baseline_clip_recall_rank']:+.3f}",
            delta_color="inverse",
        )
        st.metric(
            "Candidate Summary MRR",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_mrr']:.3f}",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_mrr'] - data['metrics']['clip_recall']['baseline_clip_recall_mrr']:+.3f}",
        )
        st.metric(
            "Candidate Summary Recall@1",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_at_1']:.3f}",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_at_1'] - data['metrics']['clip_recall']['baseline_clip_recall_at_1']:+.3f}",
        )
        st.metric(
            "Candidate Summary Recall@5",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_at_5']:.3f}",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_at_5'] - data['metrics']['clip_recall']['baseline_clip_recall_at_5']:+.3f}",
        )
        st.metric(
            "Candidate Summary 100% Recall",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_max_rank']:.3f}",
            f"{data['metrics']['clip_recall']['candidate_summary_clip_recall_max_rank'] - data['metrics']['clip_recall']['baseline_clip_recall_max_rank']:+.3f}",
            delta_color="inverse",
        )

    with c2:
        st.metric(
            "Reference Summary Rank",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_rank']:.3f}",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_rank'] - data['metrics']['clip_recall']['baseline_clip_recall_rank']:+.3f}",
            delta_color="inverse",
        )
        st.metric(
            "Reference Summary MRR",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_mrr']:.3f}",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_mrr'] - data['metrics']['clip_recall']['baseline_clip_recall_mrr']:+.3f}",
        )
        st.metric(
            "Reference Summary Recall@1",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_at_1']:.3f}",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_at_1'] - data['metrics']['clip_recall']['baseline_clip_recall_at_1']:+.3f}",
        )
        st.metric(
            "Reference Summary Recall@5",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_at_5']:.3f}",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_at_5'] - data['metrics']['clip_recall']['baseline_clip_recall_at_5']:+.3f}",
        )
        st.metric(
            "Reference Summary 100% Recall",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_max_rank']:.3f}",
            f"{data['metrics']['clip_recall']['reference_summary_clip_recall_max_rank'] - data['metrics']['clip_recall']['baseline_clip_recall_max_rank']:+.3f}",
            delta_color="inverse",
        )
    with c3:
        st.metric("Baseline Summary Rank", f"{data['metrics']['clip_recall']['baseline_clip_recall_rank']:.3f}")
        st.metric("Baseline Summary MRR", f"{data['metrics']['clip_recall']['baseline_clip_recall_mrr']:.3f}")
        st.metric("Baseline Summary Recall@1", f"{data['metrics']['clip_recall']['baseline_clip_recall_at_1']:.3f}")
        st.metric("Baseline Summary Recall@5", f"{data['metrics']['clip_recall']['baseline_clip_recall_at_5']:.3f}")
        st.metric(
            "Baseline Summary 100% Recall", f"{data['metrics']['clip_recall']['baseline_clip_recall_max_rank']:.3f}"
        )


with st.expander("Noun/Verb Recall Metrics"):
    st.subheader("Noun/Verb Recall")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Exact Cand. Summary Noun Recall",
            f"{data['metrics']['content_recall']['candidate_summary_noun_recall']:.3f}",
            f"{data['metrics']['content_recall']['candidate_summary_noun_recall'] - data['metrics']['content_recall']['baseline_noun_recall']:+.3f}",
        )
        st.metric(
            "Fuzzy Cand. Summary Noun Recall",
            f"{data['metrics']['content_recall']['candidate_summary_noun_fuzzy_recall']:.3f}",
            f"{data['metrics']['content_recall']['candidate_summary_noun_fuzzy_recall'] - data['metrics']['content_recall']['baseline_noun_fuzzy_recall']:+.3f}",
        )
        st.metric(
            "Exact Cand. Summary Verb Recall",
            f"{data['metrics']['content_recall']['candidate_summary_verb_recall']:.3f}",
            f"{data['metrics']['content_recall']['candidate_summary_verb_recall'] - data['metrics']['content_recall']['baseline_verb_recall']:+.3f}",
        )
        st.metric(
            "Fuzzy Cand. Summary Verb Recall",
            f"{data['metrics']['content_recall']['candidate_summary_verb_fuzzy_recall']:.3f}",
            f"{data['metrics']['content_recall']['candidate_summary_verb_fuzzy_recall'] - data['metrics']['content_recall']['baseline_verb_fuzzy_recall']:+.3f}",
        )
    with c2:
        st.metric(
            "Reference Summary Noun Recall",
            f"{data['metrics']['content_recall']['reference_summary_noun_recall']:.3f}",
            f"{data['metrics']['content_recall']['reference_summary_noun_recall'] - data['metrics']['content_recall']['candidate_summary_noun_recall']:+.3f}",
        )
        st.metric(
            "Reference Summary Fuzzy Noun Recall",
            f"{data['metrics']['content_recall']['reference_summary_noun_fuzzy_recall']:.3f}",
            f"{data['metrics']['content_recall']['reference_summary_noun_fuzzy_recall'] - data['metrics']['content_recall']['candidate_summary_noun_fuzzy_recall']:+.3f}",
        )
        st.metric(
            "Reference Summary Verb Recall",
            f"{data['metrics']['content_recall']['reference_summary_verb_recall']:.3f}",
            f"{data['metrics']['content_recall']['reference_summary_verb_recall'] - data['metrics']['content_recall']['candidate_summary_verb_recall']:+.3f}",
        )
        st.metric(
            "Reference Summary Fuzzy Verb Recall",
            f"{data['metrics']['content_recall']['reference_summary_verb_fuzzy_recall']:.3f}",
            f"{data['metrics']['content_recall']['reference_summary_verb_fuzzy_recall'] - data['metrics']['content_recall']['candidate_summary_verb_fuzzy_recall']:+.3f}",
        )

    with c3:
        st.metric(
            "Exact Baseline Summary Noun Recall",
            f"{data['metrics']['content_recall']['baseline_noun_recall']:.3f}",
        )
        st.metric(
            "Fuzzy Baseline Summary Noun Recall",
            f"{data['metrics']['content_recall']['baseline_noun_fuzzy_recall']:.3f}",
        )
        st.metric(
            "Exact Baseline Summary Verb Recall",
            f"{data['metrics']['content_recall']['baseline_verb_recall']:.3f}",
        )
        st.metric(
            "Fuzzy Baseline Summary Verb Recall",
            f"{data['metrics']['content_recall']['baseline_verb_fuzzy_recall']:.3f}",
        )

with st.expander("Caption Lengths"):

    # Count the number of commas and visualize with altair
    st.write("")
    df = pd.DataFrame(
        {
            "Candidate Summary": [s["candidate_summary"].count(",") for s in data["samples"]],
            "Reference Summary": [s["reference_summary"].count(",") for s in data["samples"]],
            "Baseline": [s["baseline"].count(",") for s in data["samples"]],
        }
    )
    st.altair_chart(
        alt.Chart(df)
        .transform_fold(
            ["Candidate Summary", "Reference Summary", "Baseline"],
            as_=["Caption Type", "Number of Commas"],
        )
        .mark_bar(
            opacity=0.6,
            binSpacing=0,
        )
        .encode(
            x=alt.X("Number of Commas:Q", bin=alt.Bin(maxbins=20)),
            y=alt.Y("count()", stack=None),
            color=alt.Color("Caption Type:N"),
        )
        .properties(
            title="Number of Commas",
        ),
        use_container_width=True,
    )

    # Count the length of the captions and visualize with altair
    st.write("")
    df = pd.DataFrame(
        {
            "Candidate Summary": [len(s["candidate_summary"]) for s in data["samples"]],
            "Reference Summary": [len(s["reference_summary"]) for s in data["samples"]],
            "Baseline": [len(s["baseline"]) for s in data["samples"]],
        }
    )
    st.altair_chart(
        alt.Chart(df)
        .transform_fold(
            ["Candidate Summary", "Reference Summary", "Baseline"],
            as_=["Caption Type", "Caption Length"],
        )
        .mark_bar(
            opacity=0.6,
            binSpacing=0,
        )
        .encode(
            x=alt.X("Caption Length:Q", bin=alt.Bin(maxbins=20)),
            y=alt.Y("count()", stack=None),
            color=alt.Color("Caption Type:N"),
        )
        .properties(
            title="Caption Lengths",
        ),
        use_container_width=True,
    )

    # Count the number of words and visualize with altair
    st.write("")
    df = pd.DataFrame(
        {
            "Candidate Summary": [len(s["candidate_summary"].split()) for s in data["samples"]],
            "Reference Summary": [len(s["reference_summary"].split()) for s in data["samples"]],
            "Baseline": [len(s["baseline"].split()) for s in data["samples"]],
        }
    )
    st.altair_chart(
        alt.Chart(df)
        .transform_fold(
            ["Candidate Summary", "Reference Summary", "Baseline"],
            as_=["Caption Type", "Number of Words"],
        )
        .mark_bar(
            opacity=0.6,
            binSpacing=0,
        )
        .encode(
            x=alt.X("Number of Words:Q", bin=alt.Bin(maxbins=20)),
            y=alt.Y("count()", stack=None),
            color=alt.Color("Caption Type:N"),
        )
        .properties(
            title="Number of Words",
        ),
        use_container_width=True,
    )


# Display global measures of the data
st.header("Full Results")


score_lambdas = {
    "Candidate Summary BLEU-1": lambda x: x["scores"]["candidate_summary_bleu_1"],
    "Candidate Summary BLEU-2": lambda x: x["scores"]["candidate_summary_bleu_2"],
    "Candidate Summary BLEU-3": lambda x: x["scores"]["candidate_summary_bleu_3"],
    "Candidate Summary BLEU-4": lambda x: x["scores"]["candidate_summary_bleu_4"],
    "Candidate Summary ROUGE-L": lambda x: x["scores"]["candidate_summary_rouge"],
    "Candidate Summary CIDEr": lambda x: x["scores"]["candidate_summary_cider"],
    "Candidate Summary CLIP Recall Rank": lambda x: x["scores"]["candidate_summary_clip_recall_rank"],
    "Candidate Summary CLIP MRR": lambda x: x["scores"]["candidate_summary_clip_recall_mrr"],
    "Candidate Summary Content Recall (Noun)": lambda x: x["metrics"]["content_recall"][
        "candidate_summary_noun_recall"
    ],
    "Candidate Summary Content Recall (Verb)": lambda x: x["metrics"]["content_recall"][
        "candidate_summary_verb_recall"
    ],
    "Candidate Summary Content Fuzzy Recall (Noun)": lambda x: x["metrics"]["content_recall"][
        "candidate_summary_noun_fuzzy_recall"
    ],
    "Candidate Summary Content Fuzzy Recall (Verb)": lambda x: x["metrics"]["content_recall"][
        "candidate_summary_verb_fuzzy_recall"
    ],
    "Reference Summary BLEU-1": lambda x: x["scores"]["reference_summary_bleu_1"],
    "Reference Summary BLEU-2": lambda x: x["scores"]["reference_summary_bleu_2"],
    "Reference Summary BLEU-3": lambda x: x["scores"]["reference_summary_bleu_3"],
    "Reference Summary BLEU-4": lambda x: x["scores"]["reference_summary_bleu_4"],
    "Reference Summary ROUGE-L": lambda x: x["scores"]["reference_summary_rouge"],
    "Reference Summary CIDEr": lambda x: x["scores"]["reference_summary_cider"],
    "Reference Summary CLIP Recall Rank": lambda x: x["scores"]["reference_summary_clip_recall_rank"],
    "Reference Summary CLIP MRR": lambda x: x["scores"]["reference_summary_clip_recall_mrr"],
    "Reference Summary Content Recall (Noun)": lambda x: x["metrics"]["content_recall"][
        "reference_summary_noun_recall"
    ],
    "Reference Summary Content Recall (Verb)": lambda x: x["metrics"]["content_recall"][
        "reference_summary_verb_recall"
    ],
    "Reference Summary Content Fuzzy Recall (Noun)": lambda x: x["metrics"]["content_recall"][
        "reference_summary_noun_fuzzy_recall"
    ],
    "Reference Summary Content Fuzzy Recall (Verb)": lambda x: x["metrics"]["content_recall"][
        "reference_summary_verb_fuzzy_recall"
    ],
    "Baseline Summary BLEU-1": lambda x: x["scores"]["baseline_bleu_1"],
    "Baseline Summary BLEU-2": lambda x: x["scores"]["baseline_bleu_2"],
    "Baseline Summary BLEU-3": lambda x: x["scores"]["baseline_bleu_3"],
    "Baseline Summary BLEU-4": lambda x: x["scores"]["baseline_bleu_4"],
    "Baseline Summary ROUGE-L": lambda x: x["scores"]["baseline_rouge"],
    "Baseline Summary CIDEr": lambda x: x["scores"]["baseline_cider"],
    "Baseline Summary CLIP Recall Rank": lambda x: x["scores"]["baseline_clip_recall_rank"],
    "Baseline Summary CLIP MRR": lambda x: x["scores"]["baseline_clip_recall_mrr"],
    "Baseline Summary Content Recall (Noun)": lambda x: x["metrics"]["content_recall"]["baseline_noun_recall"],
    "Baseline Summary Content Recall (Verb)": lambda x: x["metrics"]["content_recall"]["baseline_verb_recall"],
    "Baseline Summary Content Fuzzy Recall (Noun)": lambda x: x["metrics"]["content_recall"][
        "baseline_noun_fuzzy_recall"
    ],
    "Baseline Summary Content Fuzzy Recall (Verb)": lambda x: x["metrics"]["content_recall"][
        "baseline_verb_fuzzy_recall"
    ],
}

# Selection box for score sorting
score = st.selectbox("Select a score to sort by", list(score_lambdas.keys()))
# Increasing or decreasing
sort_order = st.selectbox("Select a sort order", ["Increasing", "Decreasing"])

sorted_captions = sorted(generated_captions, key=score_lambdas[score], reverse=sort_order == "Decreasing")

# Spacer
st.write("")
st.write("")

for elem in sorted_captions:
    with st.expander(f'COCO val {elem["image_path"]}'):
        with st.container():
            st.subheader(f"Image: {elem['image_path']}")
            st.image(
                Image.open(os.path.join("/ssd/coco/coco_val2014/", elem["image_path"])),
            )
            st.write(f"**Generated Candidate Summary**: _{elem['candidate_summary']}_")
            st.write(f"**Generated Reference Summary**: _{elem['reference_summary']}_")
            st.write(f"**Baseline Candidate**: _{elem['baseline']}_")

            c1, c2 = st.columns(2)
            with c1:
                st.write("Model Samples")
                st.json(elem["candidates"], expanded=False)

            with c2:
                st.write("References")
                st.json(elem["references"], expanded=False)

            st.subheader("Metrics")
            # Write a st.metric for each metric in a grid
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Cand. Sum. BLEU-1", f'{elem["scores"]["candidate_summary_bleu_1"]:0.3f}')
                st.metric("Cand. Sum. BLEU-2", f'{elem["scores"]["candidate_summary_bleu_2"]:0.3f}')
                st.metric("Cand. Sum. BLEU-3", f'{elem["scores"]["candidate_summary_bleu_3"]:0.3f}')
                st.metric("Cand. Sum. BLEU-4", f'{elem["scores"]["candidate_summary_bleu_4"]:0.3f}')
                st.metric("Cand. Sum. ROUGE-L", f'{elem["scores"]["candidate_summary_rouge"]:0.3f}')
                st.metric("Cand. Sum. CIDEr", f'{elem["scores"]["candidate_summary_cider"]:0.3f}')
                st.metric("Cand. Sum. CLIP Recall Rank", f'{elem["scores"]["candidate_summary_clip_recall_rank"]:0.3f}')
                st.metric("Cand. Sum. CLIP MRR", f'{elem["scores"]["candidate_summary_clip_recall_mrr"]:0.3f}')
                st.metric(
                    "Cand. Sum. Content Recall (Noun)",
                    f'{elem["scores"]["content_recall"]["candidate_summary_noun_recall"]:0.3f}',
                )
                st.metric(
                    "Cand. Sum. Content Recall (Verb)",
                    f'{elem["scores"]["content_recall"]["candidate_summary_verb_recall"]:0.3f}',
                )
                st.metric(
                    "Cand. Sum. Content Fuzzy Recall (Noun)",
                    f'{elem["scores"]["content_recall"]["candidate_summary_noun_fuzzy_recall"]:0.3f}',
                )
                st.metric(
                    "Cand. Sum. Content Fuzzy Recall (Verb)",
                    f'{elem["scores"]["content_recall"]["candidate_summary_verb_fuzzy_recall"]:0.3f}',
                )
            with c2:
                st.metric("Ref. Sum. BLEU-1", f'{elem["scores"]["reference_summary_bleu_1"]:0.3f}')
                st.metric("Ref. Sum. BLEU-2", f'{elem["scores"]["reference_summary_bleu_2"]:0.3f}')
                st.metric("Ref. Sum. BLEU-3", f'{elem["scores"]["reference_summary_bleu_3"]:0.3f}')
                st.metric("Ref. Sum. BLEU-4", f'{elem["scores"]["reference_summary_bleu_4"]:0.3f}')
                st.metric("Ref. Sum. ROUGE-L", f'{elem["scores"]["reference_summary_rouge"]:0.3f}')
                st.metric("Ref. Sum. CIDEr", f'{elem["scores"]["reference_summary_cider"]:0.3f}')
                st.metric("Ref. Sum. CLIP Recall Rank", f'{elem["scores"]["reference_summary_clip_recall_rank"]:0.3f}')
                st.metric("Ref. Sum. CLIP MRR", f'{elem["scores"]["reference_summary_clip_recall_mrr"]:0.3f}')
                st.metric(
                    "Ref. Sum. Content Recall (Noun)",
                    f'{elem["scores"]["content_recall"]["reference_summary_noun_recall"]:0.3f}',
                )
                st.metric(
                    "Ref. Sum. Content Recall (Verb)",
                    f'{elem["scores"]["content_recall"]["reference_summary_verb_recall"]:0.3f}',
                )
                st.metric(
                    "Ref. Sum. Content Fuzzy Recall (Noun)",
                    f'{elem["scores"]["content_recall"]["reference_summary_noun_fuzzy_recall"]:0.3f}',
                )
                st.metric(
                    "Ref. Sum. Content Fuzzy Recall (Verb)",
                    f'{elem["scores"]["content_recall"]["reference_summary_verb_fuzzy_recall"]:0.3f}',
                )
            with c3:
                st.metric("Baseline BLEU-1", f'{elem["scores"]["baseline_bleu_1"]:0.3f}')
                st.metric("Baseline BLEU-2", f'{elem["scores"]["baseline_bleu_2"]:0.3f}')
                st.metric("Baseline BLEU-3", f'{elem["scores"]["baseline_bleu_3"]:0.3f}')
                st.metric("Baseline BLEU-4", f'{elem["scores"]["baseline_bleu_4"]:0.3f}')
                st.metric("Baseline ROUGE-L", f'{elem["scores"]["baseline_rouge"]:0.3f}')
                st.metric("Baseline CIDEr", f'{elem["scores"]["baseline_cider"]:0.3f}')
                st.metric("Baseline CLIP Recall Rank", f'{elem["scores"]["baseline_clip_recall_rank"]:0.3f}')
                st.metric("Baseline CLIP MRR", f'{elem["scores"]["baseline_clip_recall_mrr"]:0.3f}')
                st.metric(
                    "Baseline Content Recall (Noun)", f'{elem["scores"]["content_recall"]["baseline_noun_recall"]:0.3f}'
                )
                st.metric(
                    "Baseline Content Recall (Verb)", f'{elem["scores"]["content_recall"]["baseline_verb_recall"]:0.3f}'
                )
                st.metric(
                    "Baseline Content Fuzzy Recall (Noun)",
                    f'{elem["scores"]["content_recall"]["baseline_noun_fuzzy_recall"]:0.3f}',
                )
                st.metric(
                    "Baseline Content Fuzzy Recall (Verb)",
                    f'{elem["scores"]["content_recall"]["baseline_verb_fuzzy_recall"]:0.3f}',
                )
