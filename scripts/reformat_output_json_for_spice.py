"""
This script reformats the output json file for the SPICE metric.
Usage:
    python scripts/reformat_output_json_for_spice.py <input_json> <output_json> <candidate key>
"""

import json
import sys


def main() -> None:
    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    output_data = []
    for i, item in enumerate(input_data):
        output_data.append(
            {
                "image_id": i,
                "test": item[sys.argv[3]],
                "refs": item["references"],
            }
        )

    with open(sys.argv[2], "w") as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    main()
