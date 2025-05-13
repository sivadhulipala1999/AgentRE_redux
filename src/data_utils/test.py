# import json
# import os

# root = os.getcwd().replace("\\", "/")
# ddir = root + "/src/data/processed_data/json"

# fn = ddir + "/test.json"

# with open(fn, encoding="utf-8") as fin:
#     file_data = json.load(fin)

# for f in file_data:
#     print(f)
#     print(file_data.get(f))
#     print("="*100)

l1 = ["Recognition", "of", "proper", "nouns", "in", "Japanese", "text", "has", "been", "studied", "as", "a", "part", "of", "the", "more", "general",
      "problem", "of", "morphological", "analysis", "in", "Japanese", "text", "processing", "-LRB-", "-LSB-", "1", "-RSB-", "-LSB-", "2", "-RSB-", "-RRB-", "."]
l2 = ["It", "has", "also", "been", "studied", "in", "the", "framework", "of", "Japanese",
      "information", "extraction", "-LRB-", "-LSB-", "3", "-RSB-", "-RRB-", "in", "recent", "years", "."]
print(l1)
