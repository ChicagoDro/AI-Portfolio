# Upgrade 03 – GraphRAG (Knowledge-Graph Retrieval)

This upgrade introduces **GraphRAG**, expanding the system beyond vector-only and hybrid (dense + BM25) retrieval into **entity-aware, relationship-aware reasoning**.

Instead of treating documents as isolated chunks, we now simulate a **knowledge graph** (similar to Neo4j) that allows the model to retrieve context based on **entities, attributes, and relationships**—not just text similarity.

This upgrade demonstrates:
- How to build a **lightweight knowledge graph layer** from an unstructured corpus  
- How to perform **neighborhood expansion**, **relationship traversal**, and **entity clustering**  
- How to **combine graph context with vector retrieval** for better grounding and precision  
- How retrieval becomes **explainable** because answers come from structured relationships

---

## 1. Goals of This Upgrade

GraphRAG solves a specific set of issues that appear when using vector or hybrid retrieval alone:

| Problem (Vector-Only / Hybrid) | How GraphRAG Helps  |
|--------------------------------|---------------------|
| Cannot reason across entities (“Which customers own bikes with upgraded suspension?”) | Graph traversal: Customers → Bikes → Upgrades |
| Cannot capture relationships | Store relationships explicitly, not implicitly in text |
| Returns text chunks but not structured knowledge | Graph returns structured entities + neighborhoods |
| Hard to explain retrieval decisions | Graph paths show *why* information is retrieved |
| Dense/BM25 retrieval works poorly for entity-centric questions | Entity lookup is exact and structured |

GraphRAG adds **structural reasoning** to your RAG architecture.

---

## 2. What We Changed in This Upgrade

### **2.1 Added a Knowledge-Graph Builder**
New file:

```

src/RAG_Chatbot/graph_kg_builder.py

```

Functions added:
- Extract entities from chunks (e.g., Customers, Bikes, Packages, Service Orders)
- Identify relationships (e.g., *purchased*, *owns*, *includes*, *installs*)
- Serialize this into a graph structure stored under:

```

indices/graph_kg.json

```

This is a **simulated Neo4j-style graph**, built entirely in Python for portability.

---

### **2.2 Added Graph-Based Retrieval Module**

New file:

```

src/RAG_Chatbot/graph_retrieval.py

```

New capabilities:
- Query-to-entity resolution  
- One-hop, two-hop, or configurable neighborhood expansion  
- Graph-context ranking (score-based weighted by:
  - node relevance  
  - relationship frequency  
  - degree centrality  
  - proximity to query intent  
)
- Fusion with vector/BM25 retrieval

This allows questions like:

> “Which packages include both suspension and exhaust upgrades?”

Traditional RAG struggles because relevant info is scattered across multiple documents.  
GraphRAG reconstructs the relationships and retrieves structured results.

---

### **2.3 Chatbot Pipeline Now Supports Graph-Aware Retrieval**

In:

```

src/RAG_Chatbot/chitown_custom_choppers_chatbot.py

````

We added:
- A **retrieval router** that determines when GraphRAG is beneficial  
- A **context fusion layer** merging:
  - graph neighbors  
  - vector hits  
  - BM25 matches  
- Toggle flags in `config.py`:
  ```python
  ENABLE_GRAPHRAG = True
  GRAPH_DEPTH = 1  # or 2+ depending on query complexity
  ```
---

### **2.4 Added Documentation + Architecture Notes**

Inside this upgrade folder, we include:

* A walkthrough of the graph-building pipeline
* Sample graph structure showing nodes + relationships
* Before/after comparisons showing why GraphRAG improves answer quality

---

## 3. Why We Made These Decisions

### **3.1 No External Database Required (Simulated Neo4j)**

We intentionally **did not** require installing Neo4j or Memgraph.

Why?

* Keep the repo **self-contained**
* Ensure that all employers/students can run the full RAG architecture locally
* Minimize environment complexity
* Make deployment easier (pure Python objects → optional future migration)

But the design **mirrors graph-database principles**:

| Graph Concept          | Our Simulation                        |
| ---------------------- | ------------------------------------- |
| Nodes                  | Entity objects extracted from corpus  |
| Relationships          | Edge list with typed relations        |
| Cypher-like queries    | Pythonic lookup + traversal functions |
| Neighborhood expansion | BFS / DFS traversals by depth         |
| Node ranking           | Weighting + scoring schema            |

This means the system can easily migrate to **actual Neo4j** later with minimal changes.

---

### **3.2 Entity-Based Retrieval Improves Accuracy**

For the Chi-Town Custom Choppers domain, many questions are **entity-centric**:

* Bikes → packages → components
* Customers → orders → builds
* Mechanics → certifications → service offerings

GraphRAG returns **structured relationships**, which improves:

* accuracy
* groundedness
* interpretability
* multi-hop reasoning

---

### **3.3 Better Query Routing**

Some examples:

| Query Type                                          | Best Method              |
| --------------------------------------------------- | ------------------------ |
| “What is the shop’s return policy?”                 | Vector / BM25            |
| “Which customers upgraded their exhaust system?”    | **GraphRAG**             |
| “What is included in the Titanium Touring Package?” | GraphRAG + vector fusion |
| “Tell me about the history of the shop.”            | Vector                   |

GraphRAG is not always best—so retrieval routing must be **intelligent**.

---

## 4. How GraphRAG Works in This Project

### **Step 1: Extract Structured Knowledge**

From unstructured text we extract:

* Entities (Customer, Bike, Package, Part, Mechanic, Service Order)
* Attributes (Year, Model, Price, Certification, etc.)
* Relationships (owns, includes, installs, purchased, performed_by, etc.)

Example snippet:

```json
{
  "nodes": {
    "Bike:Harley_FXDR": {"type": "Bike", "year": 2021},
    "Customer:John_Doe": {"type": "Customer"}
  },
  "edges": [
    {"source": "Customer:John_Doe", "target": "Bike:Harley_FXDR", "relation": "owns"},
    {"source": "Bike:Harley_FXDR", "target": "Package:PerformancePlus", "relation": "upgraded_with"}
  ]
}
```

---

### **Step 2: Query → Entities**

For each question, we attempt entity recognition:

* If user mentions **bikes, packages, customers, components**, GraphRAG activates.
* If no entities match, fallback is vector/BM25.

---

### **Step 3: Neighborhood Expansion**

We retrieve:

* Direct neighbors (depth 1)
* Or extended neighbors (depth 2–3 for more complex reasoning)

Example:

“Which packages include suspension and exhaust upgrades?”

Graph traversal finds:

* Nodes of type `Package`
* Edges linking to components of type `Suspension` or `Exhaust`
* Intersection gives the answer set

---

### **Step 4: Fuse Graph + Vector Context**

The final context window combines:

* Graph neighborhood summaries
* Supporting text chunks
* Metadata traces

This gives answers that are:

* More correct
* More explainable
* Structured and grounded

---

## 5. Demonstration Queries

Try these after enabling `ENABLE_GRAPHRAG = True`:

* **“Which bikes have both performance and touring upgrades?”**
* **“Which customers purchased the Heritage build package?”**
* **“What upgrades connect the Titan Roadster and the Iron Phantom?”**
* **“List all parts included in the Apex Performance Package.”**

Each of these relies on **relationships**, not surface text matches.

---

## 6. Future Extensions

This upgrade prepares the project for:

### 🔹 Real Neo4j integration

* Swap the simulated graph for an actual Cypher-backed DB

### 🔹 Retrieval routing agents

* Ask: “Is this question better answered by graph, vector, or hybrid?”

### 🔹 Graph-based evaluations

* Groundedness checks
* Relationship-consistency tests

### 🔹 Schema visualization

* Auto-generate entity relationship diagrams (ERDs)

---

## 7. Summary

This GraphRAG upgrade is a major architectural step:

* We **simulate a Neo4j-like graph database** for full local portability
* We added graph extraction, traversal, and hybrid fusion modules
* We improved accuracy on entity-centric questions
* We made retrieval more **explainable, structured, and robust**

This upgrade is where the system moves from “RAG demo” → **AI Solution Architecture**.

### **Portfolio Value**

You can now say:

> “I designed and integrated a GraphRAG layer on top of my existing RAG system, building a Python-based knowledge graph (Neo4j-like) to support entity-aware retrieval, graph traversal, and hybrid fusion—allowing the model to answer multi-hop, relationship-driven questions with higher precision and interpretability.”