import json
import requests
import time
import os
import wikipediaapi

# ---- CONFIG ----
OUTPUT_FILE = "data/abstracts/entity_texts_triplets.json"
SLEEP_SECONDS = 0.5  # delay between requests to avoid API rate limits
INPUT_FOLDER_PATH = 'data/entities_new/'

# ---- FUNCTIONS ----


def get_entities_from_folder(folder_path, start_idx, end_idx):
    """
    Given an input folder, read all the Wikidata JSON files and return the list of entities 
    """
    entities = dict()

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)

                    if isinstance(data, list):
                        for entry in data:
                            if entry['org1'] not in entities:
                                entities[entry['org1']] = {"triplets": [(
                                    entry['org1'], entry['relation1'], entry['midOrg'])]}
                            else:
                                entities[entry['org1']]['triplets'].append((
                                    entry['org1'], entry['relation1'], entry['midOrg']))
                            if entry['midOrg'] not in entities:
                                entities[entry['midOrg']] = {"triplets": [(
                                    entry['midOrg'], entry['relation2'], entry['org2'])]}
                            else:
                                entities[entry['org1']]['triplets'].append((
                                    entry['midOrg'], entry['relation2'], entry['org2']))

                    elif isinstance(data, dict):
                        if data['org1'] not in entities:
                            entities[data['org1']] = {"triplets": [(
                                data['org1'], data['relation1'], data['midOrg'])]}
                        else:
                            entities[data['org1']]['triplets'].append((
                                data['org1'], data['relation1'], data['midOrg']))
                        if data['midOrg'] not in entities:
                            entities[data['midOrg']] = {"triplets": [(
                                data['midOrg'], data['relation2'], data['org2'])]}
                        else:
                            entities[data['org1']]['triplets'].append((
                                data['midOrg'], data['relation2'], data['org2']))

                except json.JSONDecodeError:
                    print(f"Failed to parse JSON in file: {filename}")

    return dict(list(entities.items())[start_idx:end_idx])


def get_wikipedia_title(wikidata_id):
    """
    Given a Wikidata QID (e.g., Q95), fetch the English Wikipedia title.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    headers = {
        "User-Agent": "AgentRE_Scraper"
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        sitelink = data["entities"][wikidata_id]["sitelinks"]["enwiki"]["title"]
        return sitelink
    except KeyError:
        return None


def get_wikipedia_abstract(title):
    """
    Given a Wikipedia page title, fetch the intro summary text.
    """
    # url = "https://en.wikipedia.org/w/api.php"
    # params = {
    #     "action": "query",
    #     "format": "json",
    #     "prop": "extracts",
    #     "exintro": True,
    #     "explaintext": True,
    #     "titles": title
    # }
    # r = requests.get(url, timeout=10, params=params)
    # if r.status_code != 200:
    #     return None
    # pages = r.json().get("query", {}).get("pages", {})
    # for page_id, page in pages.items():
    #     if "extract" in page:
    #         return page["extract"]
    # return None

    # Hit the wikimedia URL to get the abstract
    wiki = wikipediaapi.Wikipedia(user_agent="AgentRE_Agent", language="en")
    page = wiki.page(title)

    # Get the summary/lead
    return page.text


# ---- MAIN ----
if __name__ == "__main__":
    # Load JSON and Collect Entity IDs
    entities = get_entities_from_folder(INPUT_FOLDER_PATH, 0, 1000)

    results = {}
    for ent in entities.items():
        # print(f"Processing {qid}...")
        qid = ent[0].split("/")[-1]
        title = get_wikipedia_title(qid)
        if title:
            abstract = get_wikipedia_abstract(title)
            results[qid] = {
                "title": title,
                "abstract": abstract,
                "triplets": ent[1]['triplets']
            }
        else:
            results[qid] = {
                "title": None,
                "abstract": None,
                "triplets": None
            }
        time.sleep(SLEEP_SECONDS)  # avoid hammering the API

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} abstracts to {OUTPUT_FILE}")
