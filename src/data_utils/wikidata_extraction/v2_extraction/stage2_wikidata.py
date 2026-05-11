import requests
import pandas as pd
import time
import json
from tqdm import tqdm


def fetch_relationships_batch(org1_batch, org2_batch, relations):
    """
    Fetch relationships for a batch of organizations
    Now batches BOTH source and target organizations
    """

    # Build VALUES clauses - both are now limited
    org1_values = ' '.join([f'wd:{org_id}' for org_id in org1_batch])
    org2_values = ' '.join([f'wd:{org_id}' for org_id in org2_batch])
    relations = ' '.join([f'{rel}' for rel in relations])

    query = f"""
SELECT DISTINCT ?org1 ?org1Label ?relLabel ?org2 ?org2Label
WHERE {{
  VALUES ?org1 {{ {org1_values} }}
  VALUES ?org2 {{ {org2_values} }}

  ?org1 ?rel ?org2 .
  ?relEntity wikibase:directClaim ?rel .

  VALUES ?rel {{
    {relations}

  }}

  FILTER(?org1 != ?org2)

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
    """

    url = 'https://query.wikidata.org/sparql'
    headers = {
        'User-Agent': 'AgentRE_Scraper',
        'Accept': 'application/sparql-results+json'
    }

    try:
        response = requests.get(
            url,
            params={'query': query, 'format': 'json'},
            headers=headers,
            timeout=90
        )
        response.raise_for_status()

        data = response.json()
        bindings = data['results']['bindings']

        relationships = []
        for binding in bindings:
            rel = {
                'source_id': binding['org1']['value'].split('/')[-1],
                'source_label': binding.get('org1Label', {}).get('value', 'Unknown'),
                'relation': binding.get('relLabel', {}).get('value', 'Unknown'),
                'target_id': binding['org2']['value'].split('/')[-1],
                'target_label': binding.get('org2Label', {}).get('value', 'Unknown')
            }
            rel['triplets'] = (
                rel['source_id'], rel['relation'], rel['target_id'])
            relationships.append(rel)

        return relationships

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 414:
            print(f"\n✗ URI too long (414). Reduce batch size further.")
        else:
            print(f"\n✗ HTTP Error {e.response.status_code}: {e}")
        return []
    except requests.exceptions.Timeout:
        print(f"\n✗ Query timeout. Consider reducing batch size.")
        return []
    except Exception as e:
        print(f"\n✗ Batch failed: {e}")
        return []


def fetch_all_relationships(org_ids, batch_size=50, relations=None):
    """
    Fetch all relationships between organizations using a grid approach

    With all the orgs, we create a grid of batch_size x batch_size queries
    This prevents URI length issues and timeouts

    Args:
        org_ids: List of Wikidata organization IDs
        batch_size: Number of orgs per batch dimension (smaller = safer)
    """

    print(f"Fetching relationships for {len(org_ids)} organizations...")
    print(f"Using batch size: {batch_size} x {batch_size}")

    # Calculate number of batches
    num_batches = (len(org_ids) + batch_size - 1) // batch_size
    total_queries = num_batches * num_batches

    print(f"Total queries to execute: {total_queries:,}")
    print(
        f"Estimated time: ~{total_queries * 2 / 60:.1f} minutes (with 2s delays)")

    all_relationships = []
    query_count = 0

    # Create batches
    org_batches = [org_ids[i:i + batch_size]
                   for i in range(0, len(org_ids), batch_size)]

    # Progress bar for outer loop
    with tqdm(total=total_queries, desc="Overall progress") as pbar:
        for i, org1_batch in enumerate(org_batches):
            for j, org2_batch in enumerate(org_batches):
                query_count += 1

                # Update progress bar description
                pbar.set_description(
                    f"Batch [{i+1}/{num_batches}] x [{j+1}/{num_batches}]")

                # Fetch relationships for this grid cell
                relationships = fetch_relationships_batch(
                    org1_batch, org2_batch, relations)
                all_relationships.extend(relationships)

                pbar.update(1)

                # Save intermediate results every 100 queries
                if query_count % 100 == 0:
                    temp_df = pd.DataFrame(all_relationships).drop_duplicates()
                    temp_df.to_csv(
                        'stage2_relationships_partial.csv', index=False)
                    print(
                        f"\n  💾 Saved {len(temp_df)} relationships so far...")

                # Rate limiting - be nice to Wikidata servers
                time.sleep(2)

    df = pd.DataFrame(all_relationships)

    if not df.empty:
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"\n✓ Found {len(df)} unique relationships")
    else:
        print("\n✗ No relationships found")

    return df


if __name__ == "__main__":
    # Load organization IDs from Stage 1
    ROOT = "src/data/Wikidata/wikidata_v2"
    try:
        with open(f'{ROOT}/org_ids_part2.json', 'r') as f:
            org_ids = json.load(f)
        print(f"Loaded {len(org_ids)} organization IDs from Stage 1")
    except FileNotFoundError:
        print("✗ Error: org_ids_part2.json not found. Please run Stage 1 first.")
        exit(1)

    # For 9616 orgs, recommend smaller batch size
    if len(org_ids) > 1000:
        print(f"\n⚠️  Large dataset detected ({len(org_ids)} orgs)")

        batch_size = 200

        print(f"\n⚠️  Considering a batch size of ({batch_size})")

    else:
        batch_size = 70

    # Option to sample for testing
    if len(org_ids) > 500:
        sample_size = 5000
        org_ids = org_ids[:sample_size]
        print(f"Using sample of {len(org_ids)} organizations")

    print("\n" + "="*60)
    print("Starting Stage 2...")
    print("="*60)

    print("Fetching all relationships...")
    relations_file_path = f"{ROOT}/relations.txt"
    with open(relations_file_path, "r") as f:
        relations = f.readlines()
        relations = [relation_desc.split(" ")[0]
                     for relation_desc in relations]

    # Fetch relationships
    rels_df = fetch_all_relationships(
        org_ids, batch_size=batch_size, relations=relations)

    if not rels_df.empty:
        # Save to CSV
        rels_df.to_csv(f'{ROOT}/stage2_relationships.csv', index=False)
        print(f"\n✓ Saved to '{ROOT}/stage2_relationships.csv'")

        # Display summary
        print("\n" + "="*60)
        print("RELATIONSHIP SUMMARY")
        print("="*60)
        print(rels_df.groupby('relation').size().sort_values(ascending=False))

        print(f"\nSample relationships:")
        print(rels_df.head(10))

        # Network statistics
        print(f"\n" + "="*60)
        print("NETWORK STATISTICS")
        print("="*60)
        print(f"Total nodes: {len(org_ids)}")
        print(f"Total edges: {len(rels_df)}")
        density = len(rels_df) / (len(org_ids) *
                                  (len(org_ids) - 1)) if len(org_ids) > 1 else 0
        print(f"Network density: {density:.6f}")

        # Clean up partial file
        import os
        if os.path.exists(f'{ROOT}/stage2_relationships_partial.csv'):
            os.remove(f'{ROOT}/stage2_relationships_partial.csv')
            print("\n✓ Cleaned up partial results file")
