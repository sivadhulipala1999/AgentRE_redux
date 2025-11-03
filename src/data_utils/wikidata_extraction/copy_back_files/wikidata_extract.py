import requests
import os
import json
import pickle

# Path to the folder
folder_path = 'data/entities_new/'

entities = []


for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)

                if isinstance(data, list):
                    for entry in data:
                        if 'org1Label' in entry:
                            entities.append(entry['org1Label'])
                        if 'midOrgLabel' in entry:
                            entities.append(entry['midOrgLabel'])
                        if 'org2Label' in entry:
                            entities.append(entry['org2Label'])

                elif isinstance(data, dict):
                    if 'org1Label' in data:
                        entities.append(data['org1Label'])
                    if 'midOrgLabel' in data:
                        entities.append(data['midOrgLabel'])
                    if 'org2Label' in data:
                        entities.append(data['org2Label'])

            except json.JSONDecodeError:
                print(f"Failed to parse JSON in file: {filename}")


def get_wikipedia_abstract(title: str, lang: str = "en") -> str:
    """
    Fetches the abstract/summary of a Wikipedia page using the REST API.

    Args:
        title (str): Title of the Wikipedia page.
        lang (str): Language code (default = 'en').

    Returns:
        str: Abstract summary text.
    """
    title = title.replace(" ", "_")

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("extract", "No abstract found.")
    except requests.RequestException as e:
        return f"Request failed: {e}"


# Example usage
entities = entities[:200]
abstracts = []

for entity in entities:
    abstract = get_wikipedia_abstract(entity)
    abstracts.append(abstract)

with open('data/abstracts/wiki_abstract_1.pkl', 'wb') as f:
    pickle.dump(abstracts, f)
