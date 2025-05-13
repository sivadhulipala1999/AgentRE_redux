import os
from easonsi import utils

root = os.getcwd().replace("\\", "/")
ddir = root + "/src/data/processed_data/json"
for fn in ['test.json', 'train.json']:
    ofn = f"{ddir}/std_{fn}"
    if os.path.exists(ofn):
        continue
    d_list = utils.LoadJson(f"{ddir}/{fn}")
    relations_dict = {"relations": []}

    for entry in d_list['examples']:
        # get the sentence and then the indices
        idx_offset = 0
        for i in range(len(entry['sentences'])):
            relations = []
            if i > 0:
                if i == 1:
                    idx_offset += len(entry['sentences'][0])
                idx_offset += len(entry['sentences'][i])
            for j in range(len(entry['relations'][i])):
                relation = {}
                head_start_idx = entry['relations'][i][j][0] - idx_offset
                head_end_idx = entry['relations'][i][j][1]+1 - idx_offset
                tail_start_idx = entry['relations'][i][j][2] - idx_offset
                tail_end_idx = entry['relations'][i][j][3]+1 - idx_offset
                relation["subject"] = {"name": " ".join(
                    entry['sentences'][i][head_start_idx:head_end_idx])}
                relation["tail"] = {"name": " ".join(
                    entry['sentences'][i][tail_start_idx:tail_end_idx])}
                relation["predicate"] = entry['relations'][i][j][4]
                relations.append(relation)
            relations_dict['relations'].append(relations)
    d_list["new_relations"] = relations_dict["relations"]
    utils.SaveJson(d_list, f"{ddir}/indra_{fn}")
