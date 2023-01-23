import json
import random

jsonl = "/ssd/flickr30k/test.jsonl"

samples = []
with open(jsonl, "r") as f:
    data = f.readlines()
    for line in data:
        samples.append(json.loads(line))

random.shuffle(samples)

# Generate a dataset
ids = set()
output_samples = []
for sample in samples:
    if sample["image_path"] in ids:
        continue
    output_samples.append({"image_path": sample["image_path"], "references": sample["meta"]["all_references"]})

with open("flickr30k_test.json", "w") as f:
    json.dump(output_samples, f, indent=4)
