import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("src/data/wikidata/wikidata_v3")
TRAIN_JSON = DATA_DIR / "std_train.json"
TEST_JSON = DATA_DIR / "std_test.json"

def get_predicate_counts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    counts = Counter()
    for entry in data:
        pred = entry["spo_list"][0]["predicate"] if entry.get("spo_list") else "UNKNOWN"
        counts[pred] += 1
    return counts

def main():
    train_counts = get_predicate_counts(TRAIN_JSON)
    test_counts = get_predicate_counts(TEST_JSON)
    
    all_predicates = sorted(set(train_counts.keys()).union(test_counts.keys()))
    
    print(f"{'Predicate':<35} | {'Train (N)':<10} | {'Train (%)':<10} | {'Test (N)':<10} | {'Test (%)':<10}")
    print("-" * 85)
    
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    train_percentages = []
    test_percentages = []
    
    for p in all_predicates:
        tr_n = train_counts.get(p, 0)
        te_n = test_counts.get(p, 0)
        tr_p = (tr_n / train_total) * 100 if train_total > 0 else 0
        te_p = (te_n / test_total) * 100 if test_total > 0 else 0
        
        train_percentages.append(tr_p)
        test_percentages.append(te_p)
        
        print(f"{p:<35} | {tr_n:<10} | {tr_p:<9.2f}% | {te_n:<10} | {te_p:<9.2f}%")
        
    # Plotting
    try:
        x = np.arange(len(all_predicates))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, train_percentages, width, label='Train %')
        rects2 = ax.bar(x + width/2, test_percentages, width, label='Test %')
        
        ax.set_ylabel('Percentage of Dataset (%)')
        ax.set_title('Predicate Distribution in Train vs Test Split')
        ax.set_xticks(x)
        ax.set_xticklabels(all_predicates, rotation=45, ha="right")
        ax.legend()
        
        fig.tight_layout()
        output_path = DATA_DIR / "split_distribution.png"
        plt.savefig(output_path)
        print(f"\nVisualization saved to {output_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    main()
