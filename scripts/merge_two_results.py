import json
import sys

with open(sys.argv[1]) as jf:
    data = json.load(jf)["samples"]

with open(sys.argv[2]) as jf:
    data2 = json.load(jf)["samples"]

outputs = []
for d in data:
    for d2 in data2:
        if d["image_path"] == d2["image_path"]:
            outputs.append(
                {
                    "image_path": d["image_path"],
                    "candidates": d["candidates"] + d2["candidates"],
                    "references": d["references"],
                    "baseline": d["baseline"],
                }
            )

with open(sys.argv[3], "w") as jf:
    json.dump({"samples": outputs}, jf, indent=4)
