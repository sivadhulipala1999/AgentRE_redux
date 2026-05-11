import json
import os


def convert_format(source_data: list) -> list:
    """
    Converts a dataset from its original QID-based format to the target SPO format.

    Args:
        source_data: A list of dictionaries, where each dictionary has a single
                     QID as its key. For example:
                     [{"Q42": {"title": ..., "abstract": ..., "triplets_readable": [...]}}]

    Returns:
        A list of dictionaries in the standardized format. For example:
        [{"text": ..., "spo_list": [{"subject": ..., "predicate": ..., "object": ...}]}]
    """
    standardized_data = []

    # Iterate over each item in the source list (e.g., {"Q42": {...}})
    for qid in source_data:
        # Since each dictionary has only one key (the QID), we extract it
        data = source_data[qid]

        # 1. Combine the title and abstract to create the 'text' field
        # We use .get() to avoid errors if a key is missing
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        combined_text = f"{title}\n\n{abstract}".strip()

        # 2. Convert the 'triplets_readable' list to the 'spo_list' format
        spo_list = []
        for readable_triplet in data.get("triplets_readable", []):
            # Ensure the triplet is valid before unpacking
            if len(readable_triplet) == 3:
                subject, predicate, obj = readable_triplet
                spo_list.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

        # 3. Append the newly formatted dictionary to our results
        standardized_data.append({
            "text": combined_text,
            "spo_list": spo_list
        })

    return standardized_data


# --- Main execution block to run the script ---
if __name__ == "__main__":
    # Define file paths
    input_filepath = "src/data/Wikidata/abstracts/entity_texts_triplets_pruned.json"
    output_filepath = "src/data/Wikidata/abstracts/std_train.json"

    # Check if the input file exists before proceeding
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at '{input_filepath}'")
    else:
        print(f"Input file found. Reading data...")
        # Load the source data from the JSON file
        with open(input_filepath, "r", encoding="utf-8") as f:
            source_data = json.load(f)

        # Perform the conversion
        print("Converting data to the standard format...")
        standardized_data = convert_format(source_data)

        # Save the new, standardized data to another JSON file
        print(f"Saving standardized data to '{output_filepath}'...")
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(standardized_data, f, indent=2, ensure_ascii=False)

        print("Conversion complete!")
