"""
Stage 1 - Wikidata SPARQL Extraction (v3)

Reads relation PIDs from relations.txt and entity-type QIDs from
entities.txt, builds an optimised SPARQL query using VALUES clauses,
and fetches organisations linked via those relations from the Wikidata
public SPARQL endpoint.

Output:
  - stage1_organisations.csv  (org_id, org_label, org_type, relation, relation_label, target_id, target_label)
  - org_ids.json              (list of unique org QIDs)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Union

import pandas as pd
import requests

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATASET_DIR = Path("src/data/wikidata/wikidata_v3")
RELATIONS_FILE = DATASET_DIR / "relations.txt"
ENTITIES_FILE = DATASET_DIR / "entities.txt"

OUTPUT_DIR = Path("src/data/wikidata/wikidata_v3")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AgentRE_redux_Scraper/3.0 (https://github.com/sivadhulipala1999/AgentRE_redux)"

# Maximum number of entity types per SPARQL batch.  Keeps each query
# lightweight enough to finish within the WDQS 60-second server limit.
BATCH_SIZE = 1
QUERY_LIMIT = 500
QUERY_RESPONSE_LIMIT = 5_000
TARGET_QID_BATCH_SIZE = 50

# ──────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────


def parse_relation_pids(filepath: Union[str, Path] = RELATIONS_FILE) -> list[str]:
    """
    Extract valid Wikidata property IDs from *relations_v2.txt*.

    Lines that contain the placeholder ``P*`` are skipped because they
    represent user-defined relations with no concrete Wikidata PID.

    Returns:
        Deduplicated list of PIDs, e.g. ``['P127', 'P355', ...]``.
    """
    pids: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            # Skip lines with the wildcard placeholder
            if "P*" in line:
                continue
            match = re.search(r"\(P(\d+)\)", line)
            if match:
                pids.append(f"P{match.group(1)}")
    # Deduplicate while preserving order
    return list(dict.fromkeys(pids))


def parse_relation_labels(filepath: Union[str, Path] = RELATIONS_FILE) -> dict[str, str]:
    """
    Build a PID → human-readable label mapping from *relations_v2.txt*.

    Each valid line has the format ``relation name (Pxxx)``.

    Returns:
        Dict mapping PIDs to labels, e.g. ``{'P127': 'parent company', ...}``.
    """
    labels: dict[str, str] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if "P*" in line:
                continue
            match = re.search(r"(.+?)\s*\(P(\d+)\)", line)
            if match:
                name = match.group(1).strip()
                pid = f"P{match.group(2)}"
                labels[pid] = name
    return labels


def parse_entity_qids(filepath: Union[str, Path] = ENTITIES_FILE) -> list[str]:
    """
    Extract Wikidata entity QIDs from *entities.txt*.

    Handles both plain ``Q123456`` and prefixed ``wd:Q123456`` formats.

    Returns:
        Deduplicated list of QIDs, e.g. ``['Q4830453', 'Q740752', ...]``.
    """
    qids: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            # Match Q-codes, optionally preceded by "wd:"
            match = re.search(r"(?:wd:)?\(?(Q\d+)\)?", line)
            if match:
                qids.append(match.group(1))
    return list(dict.fromkeys(qids))


def parse_entitytype_labels(filepath: Union[str, Path] = ENTITIES_FILE) -> dict[str, list[str]]:
    """
    Build a QID → human-readable label mapping from *entities.txt*.

    Each valid line has the format ``entity type name (Qxxx) [Level x]``.

    Returns:
        Dict mapping QIDs to labels and levels, e.g. ``{'Q4830453': ['Business', '3'], ...}``.
    """
    labels: dict[str, list[str]] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if "P*" in line:
                continue
            match = re.search(r"(.+?)\s*\(Q(\d+)\)\s*\[Level\s(\d)\]", line)
            if match:
                name = match.group(1).strip()
                qid = f"Q{match.group(2)}"
                level = f"{match.group(3)}"
                labels[qid] = [name, level]
    return labels


def get_qid_types_batch(qids: list[str], batch_size: int = 50) -> dict[str, list[str]]:
    """
    Fetches the 'instance of' (P31) QIDs for a batch of up to 50 Wikidata entities.
    Returns a dictionary mapping each input QID to a list of its type QIDs.
    """
    if not qids:
        return {}

    # Wikidata API limit is 50 entities per request
    qids_string = "|".join(qids[:batch_size])

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": qids_string,
        "props": "labels|claims",
        "languages": "en",
        "format": "json"
    }

    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(
            url, params=params, headers=headers, timeout=120).json()
        entities = response.get('entities', {})

        results = {}
        for qid in qids:
            if qid in entities:
                # P31 is 'instance of'
                claims = entities[qid].get('claims', {}).get('P31', [])
                types = [
                    c['mainsnak']['datavalue']['value']['id']
                    for c in claims if 'datavalue' in c['mainsnak']
                ]
                results[qid] = types
            else:
                results[qid] = []
        return results

    except Exception as e:
        print(f"API Error: {e}")
        return {qid: [] for qid in qids}


# ──────────────────────────────────────────────
# SPARQL query builder
# ──────────────────────────────────────────────

def build_sparql_query(
    pids: list[str],
    qids_batch: list[str],
    qids_all: list[str] = None,
    limit: int = QUERY_LIMIT,
    offset: int = 0
) -> str:
    """
    Build an optimised SPARQL query for a *batch* of entity-type QIDs.

    The query retrieves organisations whose ``wdt:P31`` (instance-of)
    value matches one of the supplied QIDs **and** that participate in
    at least one of the relations given by *pids*.

    Optimisation notes
    ------------------
    * ``VALUES`` clauses are used instead of ``FILTER(… IN (…))``
      because Blazegraph evaluates ``VALUES`` as efficient inline joins.
    * Direct ``wdt:P31`` matching is used instead of the much more
      expensive ``wdt:P31/wdt:P279*`` subclass traversal, which can
      time out when many entity types are listed.
    * Entity types are batched (see ``BATCH_SIZE``) so each individual
      query stays well within the WDQS 60-second execution limit.
    """
    type_values = " ".join(f"wd:{q}" for q in qids_batch)
    all_qids = " ".join(
        f"wd:{q}" for q in qids_all) if qids_all is not None else ""
    rel_values = " ".join(f"wdt:{p}" for p in pids)

    query = f"""\
            SELECT DISTINCT ?org ?orgLabel ?orgType ?orgTypeLabel ?rel ?target ?targetLabel
            WHERE {{
            VALUES ?orgType {{ {type_values} }}
            VALUES ?targetType {{ {all_qids} }}
            ?org wdt:P31 ?orgType .
            ?target wdt:P31 ?targetType .

            # This line ensures the entity is 'notable' and speeds up the query
            ?sitelink_org schema:about ?org ; schema:isPartOf <https://en.wikipedia.org/> .
            ?sitelink_target schema:about ?target ; schema:isPartOf <https://en.wikipedia.org/> .

            VALUES ?rel {{ {rel_values} }}
            ?org ?rel ?target .

            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            OFFSET {offset}
            LIMIT {limit}
            """

    return query


# ──────────────────────────────────────────────
# SPARQL API caller
# ──────────────────────────────────────────────

def run_sparql_query(query: str, batch_label: str = "", entity_label: str = "", relation_label: str = "") -> list[dict]:
    """
    Execute a SPARQL query against the Wikidata Query Service.

    Parameters
    ----------
    query : str
        The SPARQL query string.
    batch_label : str, optional
        A human-readable label for log messages (e.g. "batch 3/5").

    Returns
    -------
    list[dict]
        List of binding dicts from the JSON response, or an empty list
        on failure.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }

    tag = f" [{batch_label}, {entity_label}, {relation_label}]" if batch_label else ""
    print(f"⏳ Sending SPARQL query to Wikidata{tag} …")
    try:
        response = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        # Use strict=False to tolerate control characters (tabs,
        # newlines, etc.) that Wikidata labels occasionally contain.
        data = json.loads(response.text, strict=False)
        bindings = data["results"]["bindings"]
        print(f"✓ Received {len(bindings)} result bindings{tag}")
        return bindings

    except requests.exceptions.Timeout:
        print(f"✗ Query timed out (120 s){tag}. Try reducing the LIMIT.")
        return []
    except requests.exceptions.HTTPError as exc:
        print(f"✗ HTTP error{tag}: {exc}")
        return []
    except Exception as exc:
        print(f"✗ Unexpected error{tag}: {exc}")
        return []


# ──────────────────────────────────────────────
# Result processing
# ──────────────────────────────────────────────

def _extract_id(uri: str) -> str:
    """Return the trailing QID / PID from a full Wikidata URI."""
    return uri.rsplit("/", 1)[-1]


def _val(binding: dict, key: str, fallback: str = "Unknown") -> str:
    return binding.get(key, {}).get("value", fallback)


def process_results(bindings: list[dict], pid_labels: dict[str, str]) -> pd.DataFrame:
    """
    Convert raw SPARQL bindings to a tidy DataFrame and deduplicate.

    The *pid_labels* dict maps PID → human-readable name (from relations.txt)
    and is used to populate the ``relation_label`` column.

    Columns: org_id, org_label, org_type, relation, relation_label, target_id, target_label
    """
    rows = []
    for b in bindings:
        pid = _extract_id(_val(b, "rel", ""))
        rows.append(
            {
                "org_id": _extract_id(_val(b, "org", "")),
                "org_label": _val(b, "orgLabel"),
                "org_type": _val(b, "orgTypeLabel"),
                "relation": pid,
                "relation_label": pid_labels.get(pid, pid),
                "target_id": _extract_id(_val(b, "target", "")),
                "target_label": _val(b, "targetLabel"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["org_id", "relation", "target_id"])

    return df


def assign_source_target_labels(df: pd.DataFrame, qidtype_dict: dict[str, list[str]]) -> pd.DataFrame:
    """
        Reassigns appropriate labels to the source and target entities. 
        In its raw state the dataframe has multiple labels for the same source entity. Ex: Google (Q95) is labelled as an enterprise and also a public company. 
        We would like to assign the most specific label to each source entity. 

        Then, we would also like to assign the appropriate target label for the target QID, since we do not fetch that in the SPARQL query 
    """

    # Source Labels
    source_df = df[['org_id', 'org_type']]
    source_df = source_df.groupby(['org_id'])

    label_level_dict = dict(list(qidtype_dict.values()))
    source_label_dict = {}

    for org_id, group in source_df:
        org_types = group['org_type'].unique().tolist()
        ordered_group = sorted(
            list(org_types), key=lambda label: int(label_level_dict.get(label, '99')))
        source_label_dict[org_id[0]] = ordered_group[0]

    df['org_type'] = df['org_id'].map(source_label_dict)

    # Target Labels

    target_qids = df["target_id"].unique().tolist()

    qid_qidtype_dict = {}

    for i in range(0, len(target_qids), TARGET_QID_BATCH_SIZE):
        batch_qids = target_qids[i: i + TARGET_QID_BATCH_SIZE]
        types_batch = get_qid_types_batch(
            batch_qids, batch_size=TARGET_QID_BATCH_SIZE)
        qid_qidtype_dict.update(types_batch)

    dummy_str = "Irrelevant"

    def get_target_label(qids: list[str]) -> str:
        labels = []
        for qid in qids:
            label_level = qidtype_dict.get(qid, [])
            label_level = [label_level[0], int(
                label_level[1])] if label_level else []
            if not label_level:
                # Dummy entry for irrelevant entity types
                labels.append([dummy_str, 99])
            else:
                # If multiple types, join them with a comma
                labels.append(label_level)
        ordered_labels = sorted(
            labels, key=lambda x: int(x[1]))  # Sort by level
        if len(ordered_labels) == 0:
            return dummy_str
        return ordered_labels[0]

    df["target_type_ids"] = df["target_id"].map(qid_qidtype_dict)
    df["target_type"] = df["target_type_ids"].apply(get_target_label)

    print(f"Dropping {df[df['target_type'] == dummy_str].shape[0]} rows that have a target QID outside of what has been defined in the ontology")
    df = df[df['target_type'] != dummy_str]

    return df


# ──────────────────────────────────────────────
# Post-hoc stratification
# ──────────────────────────────────────────────


def stratify_dataframe(
    df: pd.DataFrame,
    target_total: int = 30_000,
) -> pd.DataFrame:
    """
    Balance the dataset by capping over-represented (org_type, relation)
    buckets via random downsampling.

    The cap per bucket is computed as ``target_total / num_non_empty_buckets``
    (floored at 50).  Buckets already below the cap are left untouched, so
    the actual total may exceed the target — but the worst outliers are
    trimmed.
    """
    group_col = ["org_type", "relation"]
    bucket_sizes = df.groupby(group_col).size()
    non_zero = bucket_sizes[bucket_sizes > 0]

    cap = max(target_total // len(non_zero), 50)

    print(f"\n─── Stratification ───")
    print(f"  Buckets: {len(non_zero)}  "
          f"(min={non_zero.min()}, median={int(non_zero.median())}, "
          f"P90={int(non_zero.quantile(0.90))}, max={non_zero.max()})")
    print(f"  Target total: {target_total:,}")
    print(f"  Cap per (org_type, relation) bucket: {cap}")

    sampled_parts = []
    trimmed = 0
    for key, group in df.groupby(group_col):
        if len(group) > cap:
            trimmed += len(group) - cap
            group = group.sample(n=cap, random_state=42)
        sampled_parts.append(group)

    result = pd.concat(sampled_parts, ignore_index=True)
    print(f"  Trimmed {trimmed:,} rows from over-represented buckets")
    print(f"  Final dataset: {len(result):,} rows (was {len(df):,})")
    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    # 1. Parse input files
    pids = parse_relation_pids()
    pid_labels = parse_relation_labels()
    qids = parse_entity_qids()
    qidtype_dict = parse_entitytype_labels()
    print(f"Loaded {len(pids)} relation PIDs: {pids}")
    print(f"PID labels: {pid_labels}")
    print(
        f"Loaded {len(qids)} entity QIDs: {qids[:5]} … ({len(qids)} total)\n")

    # 2. Batch entity types and query
    num_batches = (len(qids) + BATCH_SIZE - 1) // BATCH_SIZE
    all_bindings: list[dict] = []
    relation_set_per_batch_size = 3

    if os.path.exists(OUTPUT_DIR / "stage1_organisations_raw__.csv"):
        print(
            f"✓ Found existing output CSV at {OUTPUT_DIR / 'stage1_organisations_raw__.csv'}, skipping SPARQL query.")
        df_raw = pd.read_csv(OUTPUT_DIR / "stage1_organisations_raw__.csv")
        print(f"✓ Loaded {len(df_raw)} rows from existing CSV.")

    else:
        for i in range(0, len(qids), BATCH_SIZE):
            batch = qids[i: i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            label = f"batch {batch_num}/{num_batches}"
            offset = 0

            while QUERY_RESPONSE_LIMIT != offset:
                no_results_counter = 0
                for i in range(0, len(pids), relation_set_per_batch_size):
                    pids_batch = pids[i:i+relation_set_per_batch_size]
                    query = build_sparql_query(
                        pids=pids_batch, qids_batch=batch, qids_all=qids, offset=offset)
                    entity_label = f"entity mini-batch [{int(offset / QUERY_LIMIT + 1)}/{int(QUERY_RESPONSE_LIMIT/QUERY_LIMIT)}]"
                    relation_label = f"relation batch [{int( i / relation_set_per_batch_size + 1)}/{int(len(pids)/relation_set_per_batch_size)}]"
                    bindings = run_sparql_query(
                        query, batch_label=label, entity_label=entity_label, relation_label=relation_label)

                    if not bindings or len(bindings) == 0:
                        no_results_counter += 1
                        continue

                    all_bindings.extend(bindings)
                    time.sleep(2)

                if no_results_counter == int(len(pids)/relation_set_per_batch_size):
                    break
                offset += QUERY_LIMIT

            # Polite delay between requests to respect WDQS rate limits
            if batch_num < num_batches:
                time.sleep(2)

        if not all_bindings:
            print("No results returned across any batch - exiting.")
            return

        # 3. Process results
        df_raw = process_results(all_bindings, pid_labels)
        print(
            f"\n✓ {len(df_raw)} unique (org, relation, target) triples after dedup")

        # 4. Save outputs
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save raw (unstratified) data
        raw_csv = OUTPUT_DIR / "stage1_organisations_raw.csv"
        df_raw.to_csv(raw_csv, index=False)
        print(f"✓ Saved raw CSV  → {raw_csv}")

    # Create and save an intermediary file with source and target labels appropriately assigned
    intermediate_csv = OUTPUT_DIR / "stage1_organisations_intermediate.csv"
    df_intermediate = assign_source_target_labels(df_raw, qidtype_dict)
    df_intermediate.to_csv(intermediate_csv, index=False)
    print(
        f"✓ Saved intermediate CSV after reassigning source and target labels → {intermediate_csv}")

    # 5. Stratify to balance (org_type, relation) distribution
    df = stratify_dataframe(df_intermediate)

    csv_path = OUTPUT_DIR / "stage1_organisations.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved stratified CSV → {csv_path}")

    org_ids = df["org_id"].unique().tolist()
    json_path = OUTPUT_DIR / "org_ids.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(org_ids, f, indent=2)
    print(f"✓ Saved {len(org_ids)} unique org IDs → {json_path}")

    # 6. Summary
    print("\n─── Summary ───")
    print(df.groupby("relation").size().rename("count").to_string())
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
