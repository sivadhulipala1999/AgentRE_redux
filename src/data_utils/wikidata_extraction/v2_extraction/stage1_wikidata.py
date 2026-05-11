import os
import requests
import pandas as pd
import json


def fetch_seed_organizations(limit=100):
    """
    Fetch seed organizations from specific industries

    Args:
        industries: List of Wikidata IDs (e.g., ['Q11650', 'Q184826'])
        limit: Maximum number of organizations to fetch
    """

    query = f"""
SELECT DISTINCT ?org ?orgLabel ?countryLabel
WHERE {{
  ?org wdt:P31/wdt:P279* wd:Q4830453 .

  ?org (wdt:P127|wdt:P355|wdt:P749|wdt:P1056|wdt:P176) ?target .

  OPTIONAL {{ ?org wdt:P17 ?country . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
    """

    url = 'https://query.wikidata.org/sparql'
    headers = {
        'User-Agent': 'AgentRE_Scraper',
        'Accept': 'application/sparql-results+json'
    }

    print(f"Fetching seed organizations...")

    try:
        response = requests.get(
            url,
            params={'query': query, 'format': 'json'},
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        bindings = data['results']['bindings']

        # Parse results
        organizations = []
        for binding in bindings:
            org = {
                'id': binding['org']['value'].split('/')[-1],
                'label': binding.get('orgLabel', {}).get('value', 'Unknown'),
                'industry': binding.get('industryLabel', {}).get('value', 'Unknown'),
                'country': binding.get('countryLabel', {}).get('value', 'Unknown')
            }
            organizations.append(org)

        # Remove duplicates based on ID
        df = pd.DataFrame(organizations)
        df = df.drop_duplicates(subset=['id'])

        print(f"✓ Found {len(df)} unique organizations")

        return df

    except requests.exceptions.Timeout:
        print("✗ Query timed out. Try reducing the limit or selecting fewer industries.")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Fetch organizations
    ROOT = "src/data/Wikidata/wikidata_v2"
    orgs_df = fetch_seed_organizations(limit=40000)
    orgs_df_original = None

    if os.path.exists(f'{ROOT}/stage1_organizations.csv'):
        print(
            f"✓ Found existing file at f'{ROOT}/stage1_organizations.csv'. Loading...")
        orgs_df_original = pd.read_csv(f'{ROOT}/stage1_organizations.csv')

    if not orgs_df.empty:
        # Fetch ids not in the original file in case it exists
        if orgs_df_original is not None and not orgs_df_original.empty:
            merge_df = pd.merge(orgs_df, orgs_df_original,
                                on='id', how='outer', indicator=True)
            new_orgs_df = merge_df[merge_df['_merge'] ==
                                   'left_only'].drop_duplicates(subset=['id'])
            new_orgs_df.rename(columns={
                'label_x': 'label', 'industry_x': 'industry', 'country_x': 'country'}, inplace=True)
            new_orgs_df = new_orgs_df[['id', 'label', 'industry', 'country']]
        else:
            new_orgs_df = orgs_df

        # Save to CSV
        new_orgs_df.to_csv(
            f'{ROOT}/stage1_organizations_part2.csv', index=False)
        print(
            f"\n✓ Saved {len(new_orgs_df)} new organizations to f'{ROOT}/stage1_organizations_part2.csv'")

        # Save organization IDs for stage 2
        org_ids = new_orgs_df['id'].tolist()
        with open(f'{ROOT}/org_ids_part2.json', 'w') as f:
            json.dump(org_ids, f)
        print(f"✓ Saved organization IDs to f'{ROOT}/org_ids_part2.json'")

        # Display summary
        print("\n--- Summary ---")
        print(new_orgs_df.groupby('industry').size())
        print(f"\nSample organizations:")
        print(new_orgs_df.head(10)[['label', 'industry', 'country']])
