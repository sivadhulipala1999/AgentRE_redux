from rdflib import Graph, RDF, RDFS, OWL, URIRef
from collections import defaultdict


def analyze_ontology(turtle_file_path):
    g = Graph()
    g.parse(turtle_file_path, format="turtle")

    print(f"Loaded {len(g)} triples from {turtle_file_path}")

    # Container to hold categorized properties
    properties = defaultdict(list)

    for s, p, o in g.triples((None, RDF.type, None)):
        if o in [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
            label = g.value(subject=s, predicate=RDFS.label)
            domain = g.value(subject=s, predicate=RDFS.domain)
            range_ = g.value(subject=s, predicate=RDFS.range)

            prop_info = {
                "uri": str(s),
                "label": str(label) if label else "—",
                "domain": str(domain) if domain else "—",
                "range": str(range_) if range_ else "—",
            }

            if o == OWL.ObjectProperty:
                properties["ObjectProperty"].append(prop_info)
            elif o == OWL.DatatypeProperty:
                properties["DatatypeProperty"].append(prop_info)
            elif o == RDF.Property:
                properties["rdf:Property"].append(prop_info)

    # Print categorized properties
    for category, props in properties.items():
        print(f"\n=== {category} ({len(props)} total) ===")
        for prop in props:
            print(f"- {prop['label']} ({prop['uri']})")
            print(f"  Domain: {prop['domain']}")
            print(f"  Range:  {prop['range']}")


if __name__ == "__main__":
    # Replace with your SCORVoc TTL file path
    analyze_ontology("AgentRE_redux/src/data_utils/scor.ttl")
