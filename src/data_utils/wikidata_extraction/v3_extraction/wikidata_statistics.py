"""
Wikidata v3 Statistics

Reads stage1_organisations.csv and computes:
  1. Percentage distribution of each relation ID (PID)
  2. Percentage distribution of each entity type (org_type)

Results are printed to stdout AND saved to wikidata_stats.txt
alongside the CSV file.
"""

from __future__ import annotations

import os
import csv
from collections import Counter
from io import StringIO
from typing import Optional
import json
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data",
                        "Wikidata", "wikidata_v3")
CSV_PATH = os.path.join(DATA_DIR, "stage1_organisations.csv")
STATS_PATH = os.path.join(DATA_DIR, "wikidata_stats.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_distribution(
    title: str,
    counter: Counter,
    col1_header: str,
    col2_header: str | None = None,
    col2_map: dict[str, str] | None = None,
    width: int = 80,
) -> str:
    """Return a formatted distribution table as a string."""
    buf = StringIO()
    total = sum(counter.values())
    sorted_items = counter.most_common()

    # Column widths
    max_col1 = max(len(col1_header), *(len(k) for k, _ in sorted_items))
    if col2_header and col2_map:
        max_col2 = max(
            len(col2_header),
            *(len(col2_map.get(k, "—")) for k, _ in sorted_items),
        )
    else:
        max_col2 = 0

    buf.write("=" * width + "\n")
    buf.write(f"  {title}\n")
    buf.write("=" * width + "\n")
    buf.write(f"  Total rows : {total:,}\n")
    buf.write(f"  Unique keys: {len(sorted_items)}\n")
    buf.write("-" * width + "\n")

    # Header row
    hdr = f"  {col1_header:<{max_col1}}"
    if col2_header:
        hdr += f"  {col2_header:<{max_col2}}"
    hdr += f"  {'Count':>8}  {'Pct':>8}"
    buf.write(hdr + "\n")
    buf.write("-" * width + "\n")

    for key, count in sorted_items:
        pct = (count / total) * 100
        bar = "█" * int(pct)
        line = f"  {key:<{max_col1}}"
        if col2_map is not None:
            line += f"  {col2_map.get(key, '—'):<{max_col2}}"
        line += f"  {count:>8,}  {pct:>7.2f}%  {bar}"
        buf.write(line + "\n")

    buf.write("-" * width + "\n\n")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(csv_path: str, stats_path: str, use_percentages: bool = False) -> None:
    '''
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    relation_counter: Counter = Counter()
    entity_type_counter: Counter = Counter()
    relation_label_map: dict[str, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["relation"]
            relation_counter[pid] += 1
            entity_type_counter[row["org_type"]] += 1
            # Build label map from the CSV's own relation_label column
            if pid not in relation_label_map and "relation_label" in row:
                relation_label_map[pid] = row["relation_label"]

    if not relation_counter:
        print("No data rows found in the CSV.")
        return

    # ── Build output ─────────────────────────────────────────────────────
    output = ""
    output += _format_distribution(
        title="Relation Distribution (stage1_organisations.csv)",
        counter=relation_counter,
        col1_header="Relation ID",
        col2_header="Label",
        col2_map=relation_label_map,
    )
    output += _format_distribution(
        title="Entity-Type Distribution (stage1_organisations.csv)",
        counter=entity_type_counter,
        col1_header="Entity Type",
    )

    # Print to console
    print(output)

    # Write to file
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"[INFO] Stats saved to {stats_path}")
    '''

    train_path = os.path.join(DATA_DIR, "std_train.json")
    test_path = os.path.join(DATA_DIR, "std_test.json")

    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        print(f"[ERROR] Train or test JSON not found at {DATA_DIR}")
        return

    print("Loading train data...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    print("Loading test data...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_counts = Counter()
    for item in train_data:
        for spo in item.get("spo_list", []):
            train_counts[spo.get("predicate")] += 1

    test_counts = Counter()
    for item in test_data:
        for spo in item.get("spo_list", []):
            test_counts[spo.get("predicate")] += 1

    all_relations = set(train_counts.keys()).union(set(test_counts.keys()))
    sorted_relations = sorted(list(all_relations), key=lambda r: train_counts[r] + test_counts[r], reverse=True)

    train_vals = [train_counts[r] for r in sorted_relations]
    test_vals = [test_counts[r] for r in sorted_relations]

    if use_percentages:
        train_total = sum(train_vals)
        test_total = sum(test_vals)
        train_vals = [v / train_total * 100 if train_total else 0 for v in train_vals]
        test_vals = [v / test_total * 100 if test_total else 0 for v in test_vals]

    x = range(len(sorted_relations))
    width = 0.35

    x_train = [i - width/2 for i in x]
    x_test = [i + width/2 for i in x]

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x_train, train_vals, width, label='Train')
    rects2 = ax.bar(x_test, test_vals, width, label='Test')

    ylabel = 'Percentage of Samples (%)' if use_percentages else 'Number of Samples'
    ax.set_ylabel(ylabel)
    ax.set_title('Relation Distribution in Train and Test Sets')
    ax.set_xticks(list(x))
    ax.set_xticklabels(sorted_relations, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    out_img = os.path.join(DATA_DIR, "relation_distribution.png")
    plt.savefig(out_img, dpi=300)
    print(f"[INFO] Distribution image saved to {out_img}")



if __name__ == "__main__":
    compute_statistics(CSV_PATH, STATS_PATH, use_percentages=True)
