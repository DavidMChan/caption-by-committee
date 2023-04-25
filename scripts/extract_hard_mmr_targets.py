import json

with open("/home/davidchan/Projects/cbc/scripts/val_dataset_blip_1.0_15.json") as jf:
    blip_set = json.load(jf)

with open("/home/davidchan/Projects/cbc/scripts/val_dataset_hard_mmr_ofa_1.0_15.json") as jf:
    val_dataset = json.load(jf)


# Filter the BLIP set to only include the hard MMR targets
blip_set = [x for x in blip_set if x["image_path"] in set([y["image_path"] for y in val_dataset])]

output_samples = []
for sample in blip_set:
    for image_set in val_dataset:
        if sample["image_path"] == image_set["image_path"]:
            # sample['candidates'].extend(image_set['candidates'])
            # sample['candidates'] = [sample['candidates'][0], image_set['candidates'][0]] + sample['candidates'][1:6] + image_set['candidates'][1:6]
            # break
            # Remove the candidates
            output_samples.append(
                {
                    "image_path": sample["image_path"],
                    "references": sample["references"],
                }
            )

print(len(blip_set))

with open("/home/davidchan/Projects/cbc/scripts/val_dataset_hard_mmr_merged_v3.json", "w") as jf:
    json.dump(output_samples, jf, indent=4, sort_keys=True)
