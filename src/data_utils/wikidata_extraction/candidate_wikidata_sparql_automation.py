import spacy
import requests
import urllib.parse

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Step 1: Extract named entities


def extract_entities(text):
    doc = nlp(text)
    return list({ent.text for ent in doc.ents})

# Step 2: Get Wikidata Q-ID for an entity


def wikidata_id(entity, lang="en"):
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities"
        f"&search={urllib.parse.quote(entity)}&language={lang}&format=json"
    )
    res = requests.get(
        url, headers={"User-Agent": "AgentREEntityLookup/0.1"}).json()
    return res["search"][0]["id"] if res.get("search") else None


def wiki_summary(entity):
    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(entity)}"
    )
    res = requests.get(
        url, headers={"User-Agent": "AgentREEntityLookup/0.1"}).json()
    return res["extract"] if res.get("extract") else None

# Step 3: Get property-value pairs (triples) for a Q-ID


def wikidata_triples(qid, limit=10):
    endpoint = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?propLabel ?valLabel WHERE {{
      wd:{qid} ?p ?val .
      ?prop wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT {limit}
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "AgentREEntityLookup/0.1"
    }
    res = requests.get(endpoint, params={"query": query}, headers=headers)
    if res.status_code != 200:
        return []
    data = res.json()
    return [(b['propLabel']['value'], b['valLabel']['value']) for b in data['results']['bindings']]

# Step 4: Format triples for output


def format_triples(entity, triples, max_items=5):
    lines = [f"{entity} - {p}: {v}" for p, v in triples[:max_items]]
    return "\n".join(lines)

# Step 5: Complete baseline pipeline


def retrieve_baseline_facts(text):
    entities = extract_entities(text)
    all_facts = []
    # for ent in entities:
    #     qid = wikidata_id(ent)
    #     if qid:
    #         triples = wikidata_triples(qid)
    #         if triples:
    #             fact_block = format_triples(ent, triples)
    #             all_facts.append(fact_block)
    # return "\n\n".join(all_facts) if all_facts else "No facts found for extracted entities."
    for ent in entities:
        all_facts.append(ent + "---" + wiki_summary(ent))
    return "\n\n".join(all_facts) if all_facts else "No facts found for extracted entities."


# Example usage
if __name__ == "__main__":
    text = "Tim Berners-Lee created the World Wide Web in 1989."
    facts = retrieve_baseline_facts(text)
    print(f"\nFacts extracted for entities in:\n\"{text}\"\n")
    print(facts)
