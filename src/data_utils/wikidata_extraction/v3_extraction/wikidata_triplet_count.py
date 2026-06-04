import json 

file_path = "src/data/Wikidata/wikidata_v3/std_train.json"
file_path2 = "src/data/Wikidata/wikidata_v3/std_test.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(file_path2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

train_triples = 0 
test_triples = 0 
train_entities = set()
test_entities = set()

for idx, sample in enumerate(data):
    train_triples += len(sample['spo_list'])
    for spo in sample['spo_list']:
        train_entities.add(spo['subject'])
        train_entities.add(spo['object'])

for idx, sample in enumerate(data2):
    test_triples += len(sample['spo_list']) 
    for spo in sample['spo_list']:
        test_entities.add(spo['subject'])
        test_entities.add(spo['object'])

print(f"Total Train Triples: {train_triples}")
print(f"Total Test Triples: {test_triples}")
print(f"Total Unique Train Entities: {len(train_entities)}")
print(f"Total Unique Test Entities: {len(test_entities)}")
print(f"Total Combined Unique Entities: {len(train_entities | test_entities)}")

# for idx, sample in enumerate(data): 
#     print(f"idx: {idx}, word count: {len(sample['text'].split(' '))}")