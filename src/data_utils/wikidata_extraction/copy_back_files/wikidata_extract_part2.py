import requests
import re
import json

INPUT_FILE = "entity_texts_triplets"
OUTPUT_FILE = "entity_texts_triplets_abstracts"

# Example JSON
file_path = f"data/abstracts/{INPUT_FILE}.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 1: Extract all unique IDs (Qxxx or Pxxx)
id_pattern = re.compile(r'(Q\d+|P\d+)')
all_ids = set()

for obj in data.values():
    for triple in obj["triplets"]:
        for uri in triple:
            match = id_pattern.search(uri)
            if match:
                all_ids.add(match.group(1))

# Step 2: Batch request labels (max 50 IDs per request)


def get_labels(ids):
    ids_str = "|".join(ids)
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids_str}&format=json&props=labels&languages=en"
    headers = {
        "User-Agent": "AgentRE_Scraper"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    entities = r.json().get("entities", {})
    return {k: v["labels"]["en"]["value"] for k, v in entities.items() if "labels" in v and "en" in v["labels"]}


lookup = {}
ids_list = list(all_ids)
for i in range(0, len(ids_list), 50):
    batch = ids_list[i:i+50]
    lookup.update(get_labels(batch))

# Step 3: Add enriched triplets
for obj in data.values():
    enriched = []
    for triple in obj["triplets"]:
        readable = []
        for uri in triple:
            match = id_pattern.search(uri)
            if match:
                readable.append(lookup.get(match.group(1), match.group(1)))
            else:
                readable.append(uri)
        enriched.append(readable)
    obj["triplets_readable"] = enriched

# Pretty print result
with open(f"data/abstracts/{OUTPUT_FILE}.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
