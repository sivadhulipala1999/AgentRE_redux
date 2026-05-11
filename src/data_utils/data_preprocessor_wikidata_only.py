""" Preprocessor for DuIE and SciERC!

"""

from easonsi.util.leetcode import *
from easonsi import utils
from datasets import Dataset
import pandas as pd
import os
import sys
import json
import yaml

# Updates to process DuIE and SciERC datasets into standardized format as the original code does not work with the files downloaded
# Add logic to process Wikidata dataset as well


class Processor:

    def process_wikidata(self):
        root = os.getcwd().replace("\\", "/")
        ddir = root + "/src/data/Wikidata/wikidata_v3"
        labels = []
        for fn in ['std_train.json', 'std_test.json']:
            with open(f"{ddir}/{fn}", 'r', encoding='utf-8') as f:
                data = json.load(f)

                for item in data:
                    # Iterate over each triplet in the 'spo_list'
                    for triplet in item.get('spo_list', []):
                        # Get the predicate, if it exists
                        predicate = triplet.get('predicate')
                        if predicate:
                            if predicate not in labels:
                                labels.append(predicate)

        utils.SaveJson(labels, f"{ddir}/labels.json")
        schema_name_list = utils.LoadJson(f"{ddir}/labels.json")
        
        # Load ontology schema and mappings to construct type-aware schema
        yaml_path = f"{ddir}/scor_ds_ontology_schema.yaml"
        json_path = f"{ddir}/relation_predicate_mapping.json"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
            
        ontology = yaml.safe_load(yaml_content)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            
        pid_to_predicate = {'wdt:' + m['pid']: m['actual_predicate'] for m in mappings}
        
        predicate_to_types = {}
        for slot_name, slot_data in ontology.get('slots', {}).items():
            slot_uri = slot_data.get('slot_uri')
            if slot_uri and slot_uri in pid_to_predicate:
                pred = pid_to_predicate[slot_uri]
                predicate_to_types[pred] = {
                    'subject_type': slot_data.get('domain', 'Any'),
                    'object_type': slot_data.get('range', 'Any')
                }

        ds_schema = Dataset.from_dict({
            "predicate": schema_name_list
        })
        
        def f_process_wikidata_schema_mapped(schema):
            pred = schema['predicate']
            types = predicate_to_types.get(pred, {})
            return {
                "object_type": types.get("object_type", "Any"),
                "predicate": pred,
                "subject_type": types.get("subject_type", "Any")
            }
            
        ds_schema_processed = ds_schema.map(f_process_wikidata_schema_mapped)
        ds_schema_processed.to_json(
            f"{ddir}/std_schema.json", orient="records", lines=True, force_ascii=False)
        print(f"Saved to {ddir}/std_schema.json")


processor = Processor()
processor.process_wikidata()
print("Done!")
