import sys
import pickle
import time
import json
import openai
from openai import OpenAI
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample

# data_reader = ParquetReader(
#     "hf://datasets/HuggingFaceFW/fineweb/data", limit=500000)

# with open("chosen_docs_1.pkl", "rb") as f:
#     d = pickle.load(f)


# keywords
# with open('data/keywords/expanded_scor_keywords_thresh85p_new.json') as f:
#     keyword_dict = json.load(f)

with open('data/keywords/phrases_keywords.json') as f:
    keyword_dict = json.load(f)

keyword_flat_list = []

for ele in keyword_dict.keys():
    ele_keywords = keyword_dict[ele]
    keyword_flat_list.extend(ele_keywords)

###############################
# OR for a processing pipeline:
###############################

if __name__ == "__main__":

    from loguru import logger
    import io

    # Reconfigure default logger used internally
    logger.remove()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    logger.add(sys.stdout)  # Or sys.stdout

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-46", limit=5000000),
            LambdaFilter(lambda doc: any(
                word in doc.text for word in keyword_flat_list)),
            JsonlWriter("data/")
        ],
        tasks=1,
        workers=1
    )
    pipeline_exec.run()


# from huggingface_hub import snapshot_download
# folder = snapshot_download(
#     "HuggingFaceFW/fineweb",
#     repo_type="dataset",
#     local_dir="./fineweb/",
#     # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
#     allow_patterns="sample/100BT/*")


# # API configuration
# api_key = '8e29d53ff9bd440c52460896f05c2605'  # Replace with your API key
# base_url = "https://chat-ai.academiccloud.de/v1"
# model = "meta-llama-3.1-8b-instruct"  # Choose any available model

# # Start OpenAI client
# client = OpenAI(
#     api_key=api_key,
#     base_url=base_url
# )

# chosen_docs = []
# model_thoughts = []

# i = 0

# retry = 1
# for doc in data_reader():
#     words_in_text = set(doc.text.lower().split())  # crude tokenization
#     # preliminary filter, not every doc needs AI filtering
#     if set(keyword_flat_list) & words_in_text:
#         # Get response
#         try:
#             chat_completion = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                                 """
#                                     You are a document classifier tasked with identifying supply chain-related content.
#                                     After reviewing the document provided by the user, determine whether it qualifies as a supply chain document.
#                                     A document is considered supply chain-related if it discusses one or more of the following topics from the SCOR Digital Standard (SCOR-DS) model:
#                                         - Plan
#                                         - Source
#                                         - Transform (e.g., changes related to factories or production plants)
#                                         - Order
#                                         - Fulfill
#                                         - Orchestrate
#                                     Additionally, the document may also qualify if it discusses external events or broader themes that may indirectly impact the supply chain, such as:
#                                         - Mergers & Acquisitions
#                                         - Trade & Policy
#                                         - Disruptions
#                                         - Strategic Shifts
#                                         - Regulatory Compliance
#                                         - Technological Change
#                                         - Technology & Innovation
#                                         - Economic Conditions or Factors
#                                         - Environmental & Social Governance (ESG)
#                                     All topics should be understood in the context of a product’s lifecycle or operations related to a supply chain.
#                                     After analyzing the document:
#                                         - If it qualifies, respond: "Yes, this document is a supply chain document."
#                                         - If it does not, respond: "No, this is not a relevant document."
#                                 """
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": f"{doc.text}"
#                     }
#                 ]
#             )

#             if "Yes" in chat_completion.choices[0].message.content:
#                 chosen_docs.append(doc)
#                 # model_thoughts.append(chat_completion.choices[0].message.content)

#             time.sleep(6)

#             i += 1

#             if i % 100 == 0:
#                 print(f"Request {i} processed")
#         except openai.RateLimitError:
#             print("Rate limit hit.")
#             print(f"{i} requests processed via chat API")


# with open("chosen_docs.pkl", "wb") as f:
#     pickle.dump(chosen_docs, f)
# with open("model_thoughts.pkl", "wb") as f:
#     pickle.dump(model_thoughts, f)
