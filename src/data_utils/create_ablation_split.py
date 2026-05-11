import json
from pathlib import Path
import random
from collections import defaultdict

# Paths
DATA_DIR = Path("src/data/Wikidata/wikidata_v3")
TEST_JSON = DATA_DIR / "std_test.json"
ABLATION_JSON = DATA_DIR / "std_test_ablation.json"

def main() -> None:
    if not TEST_JSON.exists():
        print(f"[ERROR] Input file not found: {TEST_JSON}")
        return

    # Load std_test data
    print(f"Loading dataset from {TEST_JSON}")
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} entries.")

    # Group by first predicate for stratified split
    by_predicate = defaultdict(list)
    for entry in dataset:
        pred = entry["spo_list"][0]["predicate"] if entry.get("spo_list") else "UNKNOWN"
        by_predicate[pred].append(entry)

    target_size = 50
    total_examples = len(dataset)

    # Calculate exact allocation for ablation set
    allocations = {p: 0 for p in by_predicate}
    for p, group in by_predicate.items():
        allocations[p] = int(len(group) / total_examples * target_size)
        
    # Distribute the remaining spots to groups with highest fractional parts
    remaining = target_size - sum(allocations.values())
    if remaining > 0:
        fractional_parts = {p: (len(group) / total_examples * target_size) - allocations[p] for p, group in by_predicate.items()}
        for p in sorted(fractional_parts.keys(), key=lambda k: fractional_parts[k], reverse=True)[:remaining]:
            allocations[p] += 1

    ablation_data = []

    # Perform the split
    random.seed(42)
    for p, group in by_predicate.items():
        group_copy = list(group)
        random.shuffle(group_copy)
        n_ablation = allocations[p]
        ablation_data.extend(group_copy[:n_ablation])
        
    random.shuffle(ablation_data)
    
    # Save the ablation file
    print(f"Saving std_test_ablation.json with {len(ablation_data)} entries...")
    with open(ABLATION_JSON, "w", encoding="utf-8") as f:
        json.dump(ablation_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Ablation split complete")
    print("=" * 60)
    print(f"  Total source entries  : {len(dataset)}")
    print(f"  Ablation sample size  : {len(ablation_data)}")

if __name__ == "__main__":
    main()
