from glob import glob
import os
import pandas as pd
import requests
import time
import wikipediaapi
import json

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def get_property_labels_batch(property_ids):
    """Fetches labels for property IDs (e.g. P127 -> 'owned by')."""
    if not property_ids:
        return {}

    # Wiki API allows max 50 IDs usually, chunking is safer for large lists
    # For this script, we assume the list isn't massive or we'd add chunking logic
    ids_str = "|".join(property_ids[:50])

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": ids_str,
        "format": "json",
        "props": "labels",
        "languages": "en"
    }
    headers = {
        "User-Agent": "AgentRE_scraper"
    }
    try:
        data = requests.get(url, params=params, headers=headers).json()
        entities = data.get("entities", {})
        mapping = {}
        for pid, info in entities.items():
            label = info.get("labels", {}).get("en", {}).get("value", pid)
            mapping[pid] = label
        return mapping
    except Exception as e:
        print(f"Error fetching labels: {e}")
        return {pid: pid for pid in property_ids}


def get_wikipedia_content(title):
    """Fetches the intro text using Wikipedia-API."""
    try:
        # standard user agent format required by Wikipedia
        wiki = wikipediaapi.Wikipedia(
            user_agent="AgentRE_Research/1.0", language="en")
        page = wiki.page(title)

        if page.exists():
            return page.text
        return None
    except Exception as e:
        print(f"Error fetching content for {title}: {e}")
        return None


def check_presence(text, entity_label):
    """Simple inclusion check."""
    if not text or not entity_label:
        return False
    return entity_label.lower() in text.lower()

# ==========================================
# MAIN PIPELINE
# ==========================================


def process_pipeline(input_file, output_file, sleep_seconds, abstract_only=False):

    print("1. Loading Data...")

    df = pd.read_csv(input_file)
    # df = df.iloc[:100]

    print("2. Cleaning Relations...")

    def extract_pid(url):
        if isinstance(url, str) and "entity" in url:
            raise Exception("Unexpected entity link in relation column")
        if isinstance(url, str) and "/" in url:
            return url.split("/")[-1]
        return url

    if 'relation' in df.columns:
        df['p_code'] = df['relation'].apply(extract_pid)
    else:
        df['p_code'] = df['relation'].apply(extract_pid)

    unique_p_codes = df['p_code'].unique().tolist()
    print(f"   Dereferencing {len(unique_p_codes)} properties...")
    p_label_map = get_property_labels_batch(unique_p_codes)
    df['relation_label'] = df['p_code'].map(p_label_map)

    print("3. Grouping by Source Entity...")
    grouped = df.groupby(['source_id', 'source_label'])

    final_dataset = []
    total_groups = len(grouped)
    counter = 0

    print(f"   Processing {total_groups} unique entities...")

    for (src_id, src_label), group_data in grouped:
        counter += 1
        if counter % 100 == 0:
            print(f"   ...processed {counter}/{total_groups}")

        # A. Fetch Abstract (ONCE per group)
        abstract = get_wikipedia_content(src_label)

        if not abstract:
            continue

        # B. Collect Valid Triplets
        valid_triplets = []

        # Iterate through all potential relations for this source
        for _, row in group_data.iterrows():
            target_label = row['target_label']
            relation_lbl = row['relation_label']

            # C. Pruning Logic
            # We strictly check if the Target is mentioned in the text.
            # (The Source is assumed present since it's the article title)
            if check_presence(abstract, target_label):
                # Standard RE Format: (Subject, Relation, Object)
                triplet = (src_label, relation_lbl, target_label)
                valid_triplets.append(triplet)

        # Only add to dataset if we found at least one valid triplet
        if valid_triplets:
            final_dataset.append({
                'source_id': src_id,
                'source_label': src_label,
                'abstract': abstract,
                'triplets': valid_triplets  # This is now a LIST of tuples
            })

        time.sleep(sleep_seconds)

    # 4. Save
    # final_df = pd.DataFrame(final_dataset)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        final_dataset = existing_data + final_dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"Done! Saved {len(final_dataset)} documents to {output_file}")


if __name__ == "__main__":
    import glob

    # ==========================================
    # CONFIGURATION
    # ==========================================
    ROOT = "src/data/Wikidata/wikidata_v2"
    INPUT_FILES = glob.glob(f"{ROOT}/stage2_*.csv")
    # OUTPUT_FILE = f"{ROOT}/stage3_grouped_dataset_v2.json"
    OUTPUT_FILE = f"{ROOT}/stage3_grouped_dataset_v2.json"
    SLEEP_SECONDS = 0.5
    ABSTRACT_ONLY = True

    # ==========================================
    # MAIN EXECUTION
    # ==========================================
    for file_path in INPUT_FILES:
        process_pipeline(input_file=file_path, output_file=OUTPUT_FILE,
                         sleep_seconds=SLEEP_SECONDS, abstract_only=ABSTRACT_ONLY)

    # process_pipeline(input_file=INPUT_FILES[0], output_file=OUTPUT_FILE,
    #                  sleep_seconds=SLEEP_SECONDS, abstract_only=ABSTRACT_ONLY)
