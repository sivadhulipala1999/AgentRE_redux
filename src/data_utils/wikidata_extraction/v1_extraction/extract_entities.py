import requests
import time

ENDPOINT_URL = "https://query.wikidata.org/sparql"

HEADERS = {
    "Accept": "application/sparql-results+json",
    # Add your email here
    "User-Agent": "MySupplyChainBot/1.0"
}

QUERY = """
SELECT ?entity ?entityLabel ?typeLabel WHERE {{
  VALUES ?type {{
    wd:Q4830453     # Business
    wd:Q783794      # Company
    wd:Q891723      # Private/Public Company
    wd:Q43229       # Organization
    wd:Q163740      # Non-profit
    wd:Q327333      # Government agency
    wd:Q142071      # Manufacturer
  }}

  ?entity wdt:P31/wdt:P279* ?type.

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
LIMIT 500
OFFSET {offset}
"""


def run_query(offset=0, retries=3, wait=5):
    query = QUERY.format(offset=offset)
    for attempt in range(retries):
        try:
            response = requests.get(ENDPOINT_URL, params={
                                    'query': query}, headers=HEADERS, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(wait)
    print("All attempts failed.")
    return None


# Example: paginate over first 5 pages
for offset in range(0, 500, 100):
    print(f"\nFetching page starting at offset {offset}")
    data = run_query(offset)
    if data:
        for result in data["results"]["bindings"]:
            print(result["entityLabel"]["value"],
                  "-", result["typeLabel"]["value"])
    else:
        print("Nothing was fetched. Stopping the program now.")
        break
    time.sleep(2)  # Be polite to the server
