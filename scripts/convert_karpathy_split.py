import json


def main() -> None:
    # 1. Load the Karpathy split of the MSCOCO dataset
    with open("./dataset_coco.json", "r") as f:
        dataset = json.load(f)

    # 2. Construct a simple JSON file with the image paths and references for the validation set
    val_dataset = []
    for image in dataset["images"]:
        if image["split"] == "val":
            val_dataset.append(
                {
                    "image_path": image["filename"],
                    "references": [s["raw"] for s in image["sentences"]],
                }
            )

    # Write the dataset to a JSON file
    with open("./val_dataset.json", "w") as f:
        json.dump(val_dataset, f)


if __name__ == "__main__":
    main()
