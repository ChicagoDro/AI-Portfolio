# Upgrade 02 – Hybrid Search (Vector + BM25 Fusion)

**Goal:** Improve retrieval quality by combining dense semantic embeddings (FAISS) with sparse keyword retrieval (BM25).
This upgrade teaches you how real-world RAG systems improve recall, handle exact-match searches, and reduce dependence on embeddings alone.

This upgrade builds on **Upgrade 01 (Citations & Provenance)** and introduces a more intelligent, multi-channel retrieval layer.

---

# Why Hybrid Search?

FAISS (dense vector search) is great for:

* semantic similarity (“fork vibration issue” → disc brake guide)
* natural language questions
* conceptual queries

But FAISS struggles with:

* **exact terms** (“Q3 2024”)
* **unique identifiers** (part numbers like *CH-27* or *700x40C*)
* **keyword-heavy queries** (like “paint code 112-R”)
* **short queries** (“warranty”, “tire psi”)

BM25 (sparse keyword search) excels at these.

**Hybrid search = combining both to maximize retrieval accuracy.**

---

# What We Added in This Upgrade

This upgrade implements:

1. **BM25Retriever**
   Lightweight keyword-based retrieval from the same set of chunked documents.

2. **Dense Retrieval (FAISS)**
   Using `similarity_search_with_score`.

3. **Score Normalization**
   Because FAISS and BM25 scores are on different scales.

4. **Weighted Fusion**
   `combined_score = α * dense_score + β * bm25_score`

5. **Merged Ranking + Dedupe**
   Remove duplicate chunks and keep the best combined score.

6. **Enhanced Provenance**
   The sources now include dense, sparse, and hybrid scores.

---

# Architecture Diagram

```
                          ┌────────────────────┐
                          │   User Query        │
                          └─────────┬──────────┘
                                    │
             ┌──────────────────────┼─────────────────────────┐
             │                      │                         │
             ▼                      ▼                         ▼
    ┌────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
    │Classifier (LCEL)│     │FAISS Vector Store │     │BM25 Keyword Search  │
    └────────────────┘     └─────────┬─────────┘     └─────────┬──────────┘
                                     │                           │
                                     ▼                           ▼
                         ┌──────────────────┐      ┌─────────────────────┐
                         │ Dense Candidates │      │ Sparse Candidates    │
                         └─────────┬────────┘      └─────────┬──────────┘
                                   │                         │
                                   ▼                         ▼
                             Score Normalization        Score Normalization
                                   │                         │
                                   └──────────┬──────────────┘
                                              ▼
                                  Weighted Hybrid Fusion
                                              ▼
                                      Ranked Results
                                              ▼
                                      Context Builder
                                              ▼
                                           LLM
                                              ▼
                                  Answer + Provenance
```

---

# Code Changes (High-Level)

### ✔ Added BM25 Construction

Built during ingestion or at load time:

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(document_chunks)
```

### ✔ Hybrid Retrieval Function

```python
def hybrid_retrieve_with_scores(query: str):
    dense_results = faiss_store.similarity_search_with_score(query, k=5)
    sparse_results = bm25_retriever.get_relevant_documents(query)

    # Normalize, combine, score
    # Remove duplicates
    # Sort by combined hybrid score
```

### ✔ Combine Scores

BM25 gives higher = better.
FAISS gives lower = better (distance).
We normalize and invert where necessary.

Example heuristic:

```python
hybrid_score = 0.6 * dense_norm + 0.4 * sparse_norm
```

### ✔ Updated LCEL Chain

Retrieval node replaced:

```python
RunnableLambda(hybrid_retrieve_with_scores)
```

Everything else (classification, answer prompt, UI) remains compatible.

---

# Upgraded Provenance Format

Each result now includes:

```json
{
  "id": 1,
  "file_name": "Product_Wheel_Tire_Guide.pdf",
  "document_type": "Operations Manual",
  "page": 2,
  "dense_distance": 0.4451,
  "dense_conf": 0.692,
  "sparse_score": 14.2,
  "hybrid_score": 0.801,
  "channel": "hybrid"
}
```

This showcases multi-channel reasoning and retrieval transparency.

---

# Example Query Improvements

### Query:

**“What is the PSI for 26-inch tires?”**

* **Dense only:** May retrieve a frame manual (semantic confusion).
* **Sparse only:** May retrieve irrelevant docs containing "26" as a number.
* **Hybrid search:** Pulls **Wheel_Tire_Guide.pdf** consistently.

---

# What You Learned in This Upgrade

This upgrade demonstrates:

### **Core Retrieval Architecture Skills**

* Multi-channel retrieval
* Score calibration and normalization
* Weighted fusion heuristics
* Merging and deduping ranked lists

### **Production Practices**

* Enhancing RAG recall
* Supporting keyword-heavy queries
* Maintaining metadata and provenance
* Designing extensible retrieval layers

### **Portfolio Value**

You can now say:

> “I implemented a hybrid FAISS + BM25 retrieval pipeline with weighted fusion and provenance tracking to improve recall across semantic and keyword-heavy queries.”

This is exactly the kind of capability expected of an **AI Solution Architect**.

