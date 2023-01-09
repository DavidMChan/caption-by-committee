import json

with open('/home/davidchan/Projects/cbc/scripts/val_dataset_blip_1.0_15.json', 'r') as jf:
    blip_set = json.load(jf)

with open('/home/davidchan/Projects/cbc/scripts/val_dataset_hard_mmr_ofa_1.0_15.json', 'r') as jf:
    val_dataset = json.load(jf)


# Filter the BLIP set to only include the hard MMR targets
blip_set = [x for x in blip_set if x['image_path'] in set([y['image_path'] for y in val_dataset])]

for sample in blip_set:
    for image_set in val_dataset:
        if sample['image_path'] == image_set['image_path']:
            sample['candidates'].extend(image_set['candidates'])
            sample['candidates'] = [sample['candidates'][0], image_set['candidates'][0]] + sample['candidates'][1:6] + image_set['candidates'][1:6]
            break

print(len(blip_set))

with open('/home/davidchan/Projects/cbc/scripts/val_dataset_hard_mmr_merged_v2_1.0_15.json', 'w') as jf:
    json.dump(blip_set, jf, indent=4, sort_keys=True)
