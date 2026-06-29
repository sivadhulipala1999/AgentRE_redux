# Wikidata v3 Extraction Pipeline

This directory contains the `v3_extraction` pipeline scripts for building a standardized, text-grounded relation extraction dataset from Wikidata and Wikipedia. 

The pipeline is split into three main stages, designed to extract raw relational triples, ground them in natural language text, and format them for model training.

## Pipeline Overview

### Stage 1: SPARQL Extraction & Stratification (`stage1_wikidata.py`)
This script queries Wikidata to gather relational data and balances the distribution of relation types.
1. **Extraction:** Constructs and runs SPARQL queries against the Wikidata query service to fetch `(source_org, relation, target_entity)` triples.
2. **Type Specificity Collapsing:** Wikidata entities often have multiple "instance of" (P31) classifications. The script maps these against `entities.txt` (which contains specificity `[Level X]` rankings), sorts them, and collapses them into the single *most specific* entity type. It drops targets that do not fit the predefined ontology. We also did this primarily because a single QID had multiple labels which would be confusing. Collapsing into a single label removes this ambiguity.
3. **Stratification:** Balances the dataset by randomly downsampling over-represented `(org_type, relation)` combinations to ensure no single pattern dominates the training data.
- **Outputs:** `stage1_organisations.csv`, `org_ids.json`

### Stage 2: Text Grounding (`stage2_wikidata.py`)
This script adds natural language context to the extracted relational triples.
1. **Wikipedia Fetching:** Groups the extracted data by organization and pulls the full text of their English Wikipedia article.
2. **Entity Grounding:** Filters the triples by performing a text match. If a relation's `target_entity` is *not* mentioned anywhere in the organization's Wikipedia text, the relation is discarded.
3. **SPO Construction:** Compiles the successfully grounded triples into a Subject-Predicate-Object (`spo_list`) format alongside the source text.
- **Outputs:** `stage2_text_spo.json`

### Stage 3: Normalization & Splitting (`stage3_wikidata.py`)
This final script formats the grounded dataset for direct use in relation extraction models.
1. **Predicate Mapping:** Normalizes the raw Wikidata relation labels (e.g., `"parent company"`) into natural language predicates (e.g., `"is owned by"`) using the `relation_predicate_mapping.json`.
2. **Data Formatting:** Converts the list-based `spo_list` into a list of structured dictionaries: `{"subject": "...", "predicate": "...", "object": "..."}`.
3. **Train/Test Split:** Splits the fully processed dataset, allocating the first 1,000 entries for training and the remainder for testing.
- **Outputs:** `std_train.json`, `std_test.json`



# Note 
- As of now the labels for each and every entity is not present in the final std_train and std_test json files, meaning the model never gets to see the subject type and object type in the spo. This is a future work step and should be an easy extension to do. It was ignored in the initial iteration because we mention the entity supertypes in the prompt and examples for each supertype as a textual label, letting the model decide. Adding this information to the context can make the extraction much better. 