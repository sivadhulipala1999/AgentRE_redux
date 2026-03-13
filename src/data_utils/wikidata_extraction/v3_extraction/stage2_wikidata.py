"""
Stage 2 – Build (text, spo_list) dataset  (v3)

Reads the stratified stage1_organisations.csv, groups rows by org_id,
fetches the English Wikipedia article text for each organisation, prunes
triples whose target entity is not mentioned in the text, and saves the
final dataset as JSON.

Output:
  - stage2_text_spo.json   [{org_id, org_label, text, spo_list}, ...]
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import wikipediaapi

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = Path("src/data/wikidata/wikidata_v3")
INPUT_CSV = DATA_DIR / "stage1_organisations.csv"
OUTPUT_JSON = DATA_DIR / "stage2_text_spo.json"

USER_AGENT = "AgentRE_Research/3.0 (https://github.com/AgentRE)"
SLEEP_SECONDS = 0.5          # polite delay between Wikipedia requests
SAVE_INTERVAL = 100          # intermediate save every N orgs


# ──────────────────────────────────────────────
# Wikipedia helpers
# ──────────────────────────────────────────────

_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")


def get_wikipedia_text(title: str) -> str | None:
    """
    Fetch the full English Wikipedia article text for *title*.

    Returns ``None`` if the page does not exist or an error occurs.
    """
    try:
        page = _wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception as exc:
        print(f"  Warning: Wikipedia error for '{title}': {exc}")
        return None


def entity_mentioned(text: str, label: str) -> bool:
    """Case-insensitive check whether *label* appears in *text*."""
    if not text or not label:
        return False
    return label.lower() in text.lower()


# ──────────────────────────────────────────────
# Resume support
# ──────────────────────────────────────────────

def load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    """
    Load previously saved results, returning the list and a set of
    already-processed org_ids (so we can skip them).
    """
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done_ids = {entry["org_id"] for entry in data}
        print(f"Resuming: {len(data)} orgs already processed, skipping them.")
        return data, done_ids
    return [], set()


def save_results(output_path: Path, dataset: list[dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main() -> None:
    # 1. Load stage1 CSV
    if not INPUT_CSV.exists():
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows from {INPUT_CSV}")

    # 2. Group by org
    grouped = df.groupby(["org_id", "org_label"])
    total_orgs = len(grouped)
    print(f"Unique organisations: {total_orgs:,}\n")

    # 3. Resume support
    dataset, done_ids = load_existing(OUTPUT_JSON)

    processed = 0
    skipped_no_page = 0
    skipped_no_triples = 0

    for (org_id, org_label), group in grouped:
        # Skip already-processed orgs
        if org_id in done_ids:
            continue

        processed += 1
        if processed % 50 == 0:
            print(f"  ... processed {processed}/{total_orgs - len(done_ids)}  "
                  f"(dataset size: {len(dataset)})")

        # A. Fetch Wikipedia text (once per org)
        text = get_wikipedia_text(org_label)
        if not text:
            skipped_no_page += 1
            time.sleep(SLEEP_SECONDS)
            continue

        # B. Collect valid (text-grounded) triples
        spo_list: list[list[str]] = []
        for _, row in group.iterrows():
            target_label = str(row["target_label"])
            if entity_mentioned(text, target_label):
                spo_list.append([
                    org_label,
                    row["relation_label"],
                    target_label,
                ])

        if not spo_list:
            skipped_no_triples += 1
            time.sleep(SLEEP_SECONDS)
            continue

        dataset.append({
            "org_id": org_id,
            "org_label": org_label,
            "text": text,
            "spo_list": spo_list,
        })

        # Intermediate save
        if processed % SAVE_INTERVAL == 0:
            save_results(OUTPUT_JSON, dataset)
            print(f"  Intermediate save ({len(dataset)} entries)")

        time.sleep(SLEEP_SECONDS)

    # 4. Final save
    save_results(OUTPUT_JSON, dataset)

    print(f"\n{'=' * 60}")
    print(f"Stage 2 complete")
    print(f"{'=' * 60}")
    print(f"  Orgs with Wikipedia page + valid triples : {len(dataset):,}")
    print(f"  Orgs without Wikipedia page              : {skipped_no_page:,}")
    print(f"  Orgs with page but no grounded triples   : {skipped_no_triples:,}")
    print(f"  Output: {OUTPUT_JSON}")

    # Quick sample
    if dataset:
        sample = dataset[0]
        print(f"\nSample entry:")
        print(f"  org: {sample['org_label']}")
        print(f"  text: {sample['text'][:200]}...")
        print(f"  spo_list ({len(sample['spo_list'])} triples): {sample['spo_list'][:3]}")


if __name__ == "__main__":
    main()
