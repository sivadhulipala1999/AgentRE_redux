"""
Stage 3 – Post-process and split dataset (v3)

Reads the stage2_text_spo.json, applies predicate renaming based on
relation_predicate_mapping.json, and splits the data into std_train.json
(first 1000 entries) and std_test.json (remaining entries).

Output:
  - std_train.json
  - std_test.json
"""

import json
from pathlib import Path
import random
from collections import defaultdict

# Paths
DATA_DIR = Path("src/data/wikidata/wikidata_v3")
STAGE2_JSON = DATA_DIR / "stage2_text_spo.json"
MAPPING_JSON = DATA_DIR / "relation_predicate_mapping.json"
TRAIN_JSON = DATA_DIR / "std_train.json"
TEST_JSON = DATA_DIR / "std_test.json"

def main() -> None:
    if not MAPPING_JSON.exists():
        print(f"[ERROR] Mapping file not found: {MAPPING_JSON}")
        return
        
    if not STAGE2_JSON.exists():
        print(f"[ERROR] Stage 2 input file not found: {STAGE2_JSON}")
        return

    # 1. Load the predicate mapping
    print(f"Loading mapping from {MAPPING_JSON}")
    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    
    # mapping dict from wikidata_label -> actual_predicate
    predicate_map = {
        item["wikidata_label"]: item["actual_predicate"]
        for item in mapping_data
    }
    print(f"Loaded {len(predicate_map)} mappings.")

    # 2. Load the stage 2 data
    print(f"Loading dataset from {STAGE2_JSON}")
    with open(STAGE2_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} entries.")

    # 3. Apply the mapping to rename predicates
    processed_dataset = []
    
    for entry in dataset:
        new_spo_list = []
        for s, p, o in entry.get("spo_list", []):
            mapped_p = predicate_map.get(p, p)
            new_spo_list.append({"subject":s, "predicate":mapped_p, "object":o})
            
        processed_dataset.append({
            "org_id": entry.get("org_id"),
            "org_label": entry.get("org_label"),
            "text": entry.get("text"),
            "spo_list": new_spo_list
        })
        
    # 4. Split into std_train and std_test
    # Stratified split based on first predicate
    by_predicate = defaultdict(list)
    for entry in processed_dataset:
        pred = entry["spo_list"][0]["predicate"] if entry.get("spo_list") else "UNKNOWN"
        by_predicate[pred].append(entry)

    target_test_size = 150
    total_examples = len(processed_dataset)

    # Calculate exact allocation for test set
    allocations = {p: 0 for p in by_predicate}
    for p, group in by_predicate.items():
        allocations[p] = int(len(group) / total_examples * target_test_size)
        
    # Distribute the remaining test spots to groups with highest fractional parts
    remaining = target_test_size - sum(allocations.values())
    if remaining > 0:
        fractional_parts = {p: (len(group) / total_examples * target_test_size) - allocations[p] for p, group in by_predicate.items()}
        for p in sorted(fractional_parts.keys(), key=lambda k: fractional_parts[k], reverse=True)[:remaining]:
            allocations[p] += 1

    train_data = []
    test_data = []

    # Perform the split
    random.seed(42)
    for p, group in by_predicate.items():
        random.shuffle(group)
        n_test = allocations[p]
        test_data.extend(group[:n_test])
        train_data.extend(group[n_test:])
        
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 5. Save the train and test files
    print(f"Saving std_train.json with {len(train_data)} entries...")
    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saving std_test.json with {len(test_data)} entries...")
    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Stage 3 complete")
    print("=" * 60)
    print(f"  Total processed entries : {len(processed_dataset)}")
    print(f"  Train split (std_train) : {len(train_data)}")
    print(f"  Test split (std_test)   : {len(test_data)}")
    
    # Quick sample for verification
    if processed_dataset:
        sample = processed_dataset[0]
        print("\nSample processed entry:")
        print(f"  org: {sample['org_label']}")
        print(f"  spo_list: {sample['spo_list'][:3]}")

if __name__ == "__main__":
    main()
