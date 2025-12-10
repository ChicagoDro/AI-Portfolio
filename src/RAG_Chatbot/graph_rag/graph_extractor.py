
# Example GraphRAG Entity & Relationship Extractor
# This script demonstrates how to process Org_Chart.pdf and Employee_MoM_Sales_Q3_2024.pdf
# to produce nodes and edges for a knowledge graph.

import json

def extract_graph():
    nodes = [
        {"id":"peter_tamisin","type":"Person","name":"Peter 'Lil Dro' Tamisin","role":"CEO"},
        {"id":"rosa_martinez","type":"Person","name":"Rosa Martinez","role":"Director of Custom Builds"},
        {"id":"jonas_reed","type":"Person","name":"Jonas Reed","role":"Director of Repairs"},
        {"id":"samira_patel","type":"Person","name":"Samira Patel","role":"Director of Retail & Merch"}
    ]
    edges = [
        {"source":"rosa_martinez","target":"peter_tamisin","relation":"REPORTS_TO"},
        {"source":"jonas_reed","target":"peter_tamisin","relation":"REPORTS_TO"},
        {"source":"samira_patel","target":"peter_tamisin","relation":"REPORTS_TO"}
    ]
    return {"nodes":nodes,"edges":edges}

if __name__ == "__main__":
    graph = extract_graph()
    with open("graph_output.json","w") as f:
        json.dump(graph,f,indent=2)
