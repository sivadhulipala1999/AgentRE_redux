import pickle
import time
import json
import openai
from openai import OpenAI

# Approach 1
# with open("chosen_docs_alt.pkl", "rb") as f:
#     d = pickle.load(f)

# Approach 2
d = []
limit = 2000
with open("data/filter_pipeline_file_batch.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= limit:
            break
        d.append(json.loads(line))


# d = d[1400:2000]

api_key = '8e29d53ff9bd440c52460896f05c2605'  # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"
model = "meta-llama-3.1-8b-instruct"  # Choose any available model
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

with open('data/keywords/expanded_scor_keywords_thresh85p_new.json') as f:
    keyword_dict = json.load(f)


keyword_flat_list = []

for ele in keyword_dict.keys():
    ele_keywords = keyword_dict[ele]
    keyword_flat_list.extend(ele_keywords)

chosen_docs = []

i = 1
for doc in d:
    words_in_text = set(doc['text'].lower().split())  # crude tokenization
    # preliminary filter, not every doc needs AI filtering
    # if set(keyword_flat_list) & words_in_text:
    # Get response
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                            """
                                You are a document classifier tasked with identifying supply chain-related content.
                                After reviewing the document provided by the user, determine whether it qualifies as a supply chain document.
                                A document is considered supply chain-related if it discusses one or more of the following topics from the SCOR Digital Standard (SCOR-DS) model:
                                    - Plan
                                    - Source
                                    - Transform (e.g., changes related to factories or production plants)
                                    - Order
                                    - Fulfill
                                    - Orchestrate
                                Additionally, the document may also qualify if it discusses external events or broader themes that may indirectly impact the supply chain, such as:
                                    - Mergers & Acquisitions
                                    - Trade & Policy
                                    - Disruptions
                                    - Strategic Shifts
                                    - Regulatory Compliance
                                    - Technological Change
                                    - Technology & Innovation
                                    - Economic Conditions or Factors
                                    - Environmental & Social Governance (ESG)
                                All topics should be understood in the context of a product's lifecycle or operations related to a supply chain.
                                After analyzing the document:
                                    - If it qualifies, respond: "Yes, this document is a supply chain document."
                                    - If it does not, respond: "No, this is not a relevant document."
                            """
                    )
                },
                {
                    "role": "user",
                    "content": f"{doc['text']}"
                }
            ]
        )

        if "Yes" in chat_completion.choices[0].message.content:
            chosen_docs.append(doc)
            # model_thoughts.append(chat_completion.choices[0].message.content)

        time.sleep(6)

        if i % 100 == 0:
            print(f"Request {i} processed")

        i += 1

    except openai.RateLimitError:
        print("Rate limit hit.")
        print(f"{i} requests processed via chat API")

    except openai.APITimeoutError:
        print(f"API time out occured. Requests processed - {i}")


with open("data/chosen_docs_llm_1.pkl", "wb") as f:
    pickle.dump(chosen_docs, f)
