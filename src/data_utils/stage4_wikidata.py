"""Combine all partial files into a single CSV and visualize"""

import json
import glob

INPUT_FOLDER = "src/data/Wikidata/wikidata_v2"
# INPUT_FILES = glob.glob(f"{INPUT_FOLDER}/stage3_*.json")
# Use the combined file from Stage 3
INPUT_FILES = [f"{INPUT_FOLDER}/stage3_grouped_dataset_v2.json"]
combined_data = []

for file_path in INPUT_FILES:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        triplet_list = []
        for ent in data:
            for triplet in ent['triplets']:
                triplet_list.append({
                    "subject": triplet[0],
                    "predicate": triplet[1],
                    "object": triplet[2]
                })
            reformatted_data = {
                "text": ent["abstract"],
                "spo_list": triplet_list
            }
            combined_data.append(reformatted_data)


with open(f"{INPUT_FOLDER}/final_wikidata_dataset.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

with open(f"{INPUT_FOLDER}/std_train.json", "w", encoding="utf-8") as f:
    json.dump(combined_data[:1001], f, ensure_ascii=False, indent=2)

with open(f"{INPUT_FOLDER}/std_test.json", "w", encoding="utf-8") as f:
    json.dump(combined_data[1001:], f, ensure_ascii=False, indent=2)
