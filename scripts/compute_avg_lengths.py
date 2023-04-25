import json

import numpy as np

with open("/home/davidchan/Projects/cbc/scripts/val_dataset_hard_mmr_merged_v3_blip_1.0_10.json") as jf:
    data = json.load(jf)["samples"]

print(np.mean([len(d["candidate_summary"]) for d in data]))
print(np.mean([len(d["reference_summary"]) for d in data]))

# Count the words "likely", "probably", "possibly" in the candidate summaries
print(
    np.mean(
        [
            len([w for w in d["candidate_summary"].split() if w.strip().lower() in ["likely", "probably", "possibly"]])
            for d in data
        ]
    )
)
print(
    np.mean(
        [
            len([w for w in d["reference_summary"].split() if w.strip().lower() in ["likely", "probably", "possibly"]])
            for d in data
        ]
    )
)
