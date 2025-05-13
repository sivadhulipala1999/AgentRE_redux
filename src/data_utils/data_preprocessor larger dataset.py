""" Preprocessor for DuIE and SciERC!

"""

from easonsi.util.leetcode import *
from easonsi import utils
from datasets import Dataset
import pandas as pd
import os
import sys
import json


class Processor:

    @staticmethod
    def f_process_duie_sample(sample):
        spo_list_new = []
        for spo in sample['spo_list']:
            spo_list_new.append({
                "subject": spo['subject'],
                "predicate": spo['predicate'],
                "object": spo['object']['@value']
            })
        return {'spo_list': spo_list_new}

    @staticmethod
    def f_process_duie_schema(schema) -> None:
        schema['object_type'] = schema['object_type']['@value']
        return schema

    def process_duie(self):
        ddir = "/home/ubuntu/work/agent/AgentIE/data/DuIE2.0"
        for fn in ['duie_sample.json', 'duie_dev.json']:
            ofn = f"{ddir}/std_{fn}"
            if os.path.exists(ofn):
                continue
            # ds = Dataset.from_json(f"{ddir}/{fname}")     # has bug!!!
            df = pd.read_json(f"{ddir}/{fn}", lines=True)
            ds = Dataset.from_pandas(df)
            ds_processed = ds.map(self.f_process_duie_sample)
            ds_processed.to_json(ofn, orient="records",
                                 lines=True, force_ascii=False)
            print(f"Saved to {ofn}")
        fn_schema = f"duie_schema.json"
        ds_schema = Dataset.from_json(f"{ddir}/{fn_schema}")
        ds_schema_processed = ds_schema.map(self.f_process_duie_schema)
        ds_schema_processed.to_json(
            f"{ddir}/std_{fn_schema}", orient="records", lines=True, force_ascii=False)
        print(f"Saved to {ddir}/std_{fn_schema}")

    @staticmethod
    def f_process_scierc_sample(sample):
        # Prep a string and list of dictionaries from the example for downstream processing as per the trainer class
        final_text = ""
        spo_list_new = []
        for i in range(len(sample['text'])):
            final_text = final_text + " " + " ".join(sample['text'][i])
        for j in range(len(sample['spo_list'])):
            if sample['spo_list'][j]['head'] is None:
                spo_list_new.append(
                    {'subject': '', 'predicate': '', 'object': ''})
                continue
            spo_list_new.append({
                "subject": sample['spo_list'][j]['head']['name'],
                "predicate": sample['spo_list'][j]['type'],
                "object": sample['spo_list'][j]['tail']['name']
            })
        return {'text': final_text, 'spo_list': spo_list_new}

    @staticmethod
    def f_process_scierc_schema(schema) -> None:
        schema_new = {
            "object_type": "Any",
            "predicate": schema['predicate'],
            "subject_type": "Any"
        }
        return schema_new

    def process_sciERC(self):
        root = os.getcwd().replace("\\", "/")
        ddir = root + "/src/data/processed_data/json"
        ddir_tmp = root + "/src/data/SciERC_sample_10000"
        labels = []
        for fn in ['test__.json', 'train__.json']:
            ofn = f"{ddir_tmp}/std_{fn}"
            if os.path.exists(ofn):
                continue
            with open(f"{ddir}/{fn}") as f:
                lines = f.readlines()
                sentences = []
                new_relations = []
                for line in lines:
                    entry = json.loads(line)
                    # get the sentence and then the indices
                    idx_offset = 0
                    relations = []
                    for i in range(len(entry['sentences'])):
                        if i > 0:
                            if i == 1:
                                idx_offset += len(entry['sentences'][0])
                            idx_offset += len(entry['sentences'][i])
                        if len(entry['relations'][i]) == 0:
                            relations.append({})
                        for j in range(len(entry['relations'][i])):
                            relation = {}
                            head_start_idx = entry['relations'][i][j][0] - \
                                idx_offset
                            head_end_idx = entry['relations'][i][j][1] + \
                                1 - idx_offset
                            tail_start_idx = entry['relations'][i][j][2] - \
                                idx_offset
                            tail_end_idx = entry['relations'][i][j][3] + \
                                1 - idx_offset
                            relation["head"] = {"name": " ".join(
                                entry['sentences'][i][head_start_idx:head_end_idx])}
                            relation["tail"] = {"name": " ".join(
                                entry['sentences'][i][tail_start_idx:tail_end_idx])}
                            relation["type"] = entry['relations'][i][j][4]
                            relations.append(relation)

                            if relation["type"] not in labels:
                                labels.append(relation["type"])
                        # utils.SaveJson(d_list, f"{ddir}/indra_{fn}")
                    sentences.append(entry['sentences'])
                    new_relations.append(relations)

            dataset_dict = {
                "text": sentences,
                "spo_list": new_relations
            }

            ds = Dataset.from_dict(dataset_dict)
            ds_processed = ds.map(self.f_process_scierc_sample)
            ds_processed.to_json(ofn, orient="records",
                                 lines=False, force_ascii=False)
            print(f"Saved to {ofn}")
        utils.SaveJson(labels, f"{ddir_tmp}/labels.json")
        schema_name_list = utils.LoadJson(f"{ddir_tmp}/labels.json")
        ds_schema = Dataset.from_dict({
            "predicate": schema_name_list
        })
        ds_schema_processed = ds_schema.map(self.f_process_scierc_schema)
        ds_schema_processed.to_json(
            f"{ddir_tmp}/std_schema.json", orient="records", lines=True, force_ascii=False)
        print(f"Saved to {ddir_tmp}/std_schema.json")


processor = Processor()
# processor.process_duie()
processor.process_sciERC()
print("Done!")
