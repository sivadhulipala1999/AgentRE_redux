import copy
import numpy as np
from nltk.corpus import words
import spacy
import nltk
import json
import pickle
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Load base SCOR JSON (your file or dict)
with open('seed_keywords_gemini.json') as f:
    base_json = json.load(f)

with open('seed_keywords_deepseek.json') as f:
    base_json2 = json.load(f)

with open('seed_keywords_openai.json') as f:
    base_json3 = json.load(f)


final_json = copy.deepcopy(base_json)

for k in base_json2.keys():
    if k in base_json.keys():
        final_json[k].extend(base_json2[k])
    else:
        final_json[k] = base_json2[k]


for k in base_json3.keys():
    if k in base_json.keys():
        final_json[k].extend(base_json3[k])
    else:
        final_json[k] = base_json3[k]

# deduplication
for k in final_json.keys():
    final_json[k] = list(dict.fromkeys(s.lower() for s in final_json[k]))

# Initialize SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Download required NLTK and spaCy resources
# nltk.download('words')
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Flatten keywords
keyword_to_category = {}
all_keywords = []
for category, details in final_json.items():
    for kw in details:
        keyword_to_category[kw] = category
        all_keywords.append(kw)

# Encode original keywords
keyword_embeddings = model.encode(all_keywords, convert_to_tensor=True)

# Load or define a candidate vocabulary list (you can use a large list or scrape documents)
candidate_vocab = list(set(words.words()))
# filtered_vocab = []
# for word in candidate_vocab:
#     doc = nlp(word)
#     if doc and doc[0].pos_ in ["NOUN", "VERB"] and len(word) > 2:
#         filtered_vocab.append(word.lower())

# with open("filtered_words.pkl", "wb") as f:
#     pickle.dump(filtered_vocab, f)

with open("filtered_words.pkl", "rb") as f:
    filtered_vocab = pickle.load(f)

# Encode the new vocab
vocab_embeddings = model.encode(filtered_vocab, convert_to_tensor=True)

# Compute cosine similarities
cos_scores = util.cos_sim(keyword_embeddings, vocab_embeddings)

# Expansion threshold
THRESHOLD = 0.85

# Generate expanded JSON
expanded_json = defaultdict(lambda: {"keywords": set()})

for i, keyword in enumerate(all_keywords):
    base_category = keyword_to_category[keyword]
    expanded_json[base_category]['keywords'].add(keyword)  # include original

    for j, score in enumerate(cos_scores[i]):
        if score >= THRESHOLD:
            expanded_json[base_category]["keywords"].add(filtered_vocab[j])

# Convert sets to lists and include subcategories from original JSON
for category in expanded_json:
    expanded_json[category]["keywords"] = list(
        expanded_json[category]["keywords"])

# Save to file
with open('data/keywords/expanded_scor_keywords_thresh85p_new.json', 'w', encoding='UTF-8') as f:
    json.dump(expanded_json, f, indent=2)
