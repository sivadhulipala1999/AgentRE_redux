import json
import spacy

# load spaCy for NER (optional, just for entity detection)
nlp = spacy.load("en_core_web_md")


def prune_entry(qid, entry):
    abstract = entry["abstract"].lower()
    triplets = entry["triplets"]
    triplets_readable = entry["triplets_readable"]

    pruned_triplets = []
    pruned_triplets_readable = []

    for t, tr in zip(triplets, triplets_readable):
        head_label, rel, tail_label = tr  # from readable triplets
        # check if head or tail label appears in abstract
        if head_label.lower() in abstract and tail_label.lower() in abstract:
            pruned_triplets.append(t)
            pruned_triplets_readable.append(tr)

    return {
        "title": entry["title"],
        "abstract": entry["abstract"],
        "triplets": list(set(tuple(t) for t in pruned_triplets)),
        "triplets_readable": list(set(tuple(t) for t in pruned_triplets_readable))
    }


# load dataset
file_path = "data/abstracts/entity_texts_triplets_abstracts.json"
output_file_path = "data/abstracts/entity_texts_triplets_pruned.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# prune all entries and remove those without any relations
pruned_data = {qid: prune_entry(qid, entry) for qid, entry in data.items()}
pruned_data = {qid: entry for qid, entry in pruned_data.items()
               if entry['triplets']}

# save output
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(pruned_data, f, ensure_ascii=False, indent=2)
    print(f"{len(pruned_data)} entries recorded")
