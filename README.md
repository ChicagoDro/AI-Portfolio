# AI Portfolio – Chi-Town Custom Choppers RAG Lab

A practical GenAI portfolio project showcasing **real-world retrieval architectures**:

- Classic **vector RAG**
- **Hybrid** dense + BM25 retrieval
- **GraphRAG** over a knowledge graph
- Room for **evaluation + orchestration** upgrades

The sample dataset used in this project is a synthetic, LLM-generated knowledge base representing a fictional bicycle shop called **Chi-Town Custom Choppers**. It includes richly detailed documents covering customers, bikes, service orders, upgrade packages, parts catalogs, shop policies, mechanic bios, and narrative descriptions of custom build workflows. Although entirely artificial, the dataset is structured to mimic realistic business operations, complete with relationships between entities such as customers → bikes → upgrades, mechanics → certifications, and packages → components. This makes it ideal for demonstrating RAG, hybrid search, and GraphRAG techniques in a safe, non-sensitive environment while still providing enough complexity to showcase multi-hop reasoning, entity extraction, and retrieval orchestration.

---

## 1. What This Project Demonstrates

This repo is meant to show employers, collaborators, and students that you can:

- Design and implement **production-grade RAG systems**, not just toy demos  
- Evolve an architecture from **simple vector search → hybrid RAG → GraphRAG**
- Think like an **AI Solution Architect**: clear modules, extension points, and documentation
- Work with modern GenAI tooling (LLMs, embeddings, vector stores, knowledge graphs)

---

## 2. High-Level Architecture

At a high level, the system looks like this:

```text
          ┌─────────────────────┐
          │  Domain Documents   │  (data/Chitown_Custom_Choppers)
          └─────────┬───────────┘
                    │
        Ingestion & Indexing (offline)
                    │
     ┌──────────────┴───────────-────────────┐
     │                                       │
┌────▼──────────┐                     ┌──────▼───────────┐
│ Vector Index  │ (FAISS, embeddings) │ Knowledge Graph  │
│ (indices/...) │                     │ (entities, edges)│
└────┬──────────┘                     └──────┬───────────┘
     │                                       │
     └──────────────┬────────────--──────────┘
                    │
           Hybrid Query Engine
  (semantic + BM25 + graph-aware retrieval)
                    │
                    ▼
              LLM Answering
      (OpenAI / Grok or other providers)
````

### Retrieval Strategies

The core chatbot uses multiple retrieval modes:

1. **Vector RAG (dense)**

   * Embed chunks from the Chi-Town docs
   * Store in a FAISS (or similar) index under `indices/`
   * Retrieve semantically similar chunks for each query

2. **Hybrid Search (dense + BM25)**

   * Combine **semantic vector search** with **sparse BM25**
   * Use **fusion** (e.g., reciprocal rank fusion) to balance:

     * “semantic closeness” (meaning)
     * “lexical match” (exact terms, SKUs, part names, etc.)

3. **GraphRAG (knowledge graph–aware)**

   * Build a **knowledge graph** from the same docs:

     * Extract entities (customers, bikes, parts, work orders…)
     * Extract relationships (owns, purchased, installed, scheduled, etc.)
   * At query time, the system can:

     * Expand around query-relevant entities
     * Pull a subgraph as structured context
     * Optionally fuse graph context with vector results

4. **Hybrid Query Planner (routing + fusion)**

   * For “who/what/which” entity-centric questions, lean more on **GraphRAG**
   * For fuzzy, descriptive questions, lean more on **semantic + BM25 hybrid**
   * Combine multiple signals into a final ranked context window for the LLM

> The **goal** is to show you understand *how* to combine these pieces, not just that you can call a single `RetrievalQA` chain.

---

## 3. Repository Layout

```text
AI-Portfolio/
├── README.md                     # ← You are here
├── requirements.txt
├── .env.example                  # sample environment variables (.env is gitignored)
├── .python-version               # Python version used locally
├── main.py                       # (optional entrypoint / orchestration)
├── src/
│   └── RAG_Chatbot/
│       ├── ingest_embed_index.py         # Build vector + graph indices from data/
│       ├── chitown_custom_choppers_chatbot.py  # Main chatbot / Streamlit app
│       ├── graph_kg_builder.py          # Knowledge graph construction (GraphRAG)
│       ├── graph_retrieval.py           # Graph-based + hybrid retrieval helpers
│       └── config.py                    # Central configuration (models, paths, etc.)
├── data/
│   ├── document-metadata.json           # Chunk + document metadata
│   └── Chitown_Custom_Choppers/        # Domain docs (LLM-generated shop data)
├── indices/
│   └── faiss_chitowncustomchoppers_index/  # Generated vector + graph indices
├── notebooks/                           # Scratch space / experiments
└── upgrades/
    └── RAG_Chatbot/
        ├── 01-citations-and-provenance/
        ├── 02-hybrid-search/
        ├── 03-graph-rag/
        └── 04-evaluation-and-orchestration/
```

### `src/` – Core System

* **`ingest_embed_index.py`**

  * Loads domain docs from `data/Chitown_Custom_Choppers/`
  * Splits into chunks, embeds them, and writes a **vector index** under `indices/`
  * Optionally calls `graph_kg_builder.py` to build / refresh the **knowledge graph**

* **`chitown_custom_choppers_chatbot.py`**

  * Main chatbot application (typically launched via **Streamlit**)
  * Orchestrates:

    * user input → query analysis
    * hybrid retrieval (vector + BM25 + graph)
    * LLM invocation
    * display of answers + citations

* **`graph_kg_builder.py`**

  * Extracts entities and relationships from the corpus
  * Persists a graph structure used by GraphRAG

* **`graph_retrieval.py`**

  * Implements graph-based lookups and hybrid (graph + vector) retrieval flows

* **`config.py`**

  * Central configuration: model names, paths, index locations, and feature flags (e.g. toggle hybrid/GraphRAG behavior)

### `upgrades/` – Progressive Learning Path

Each folder is a **self-contained upgrade** that explains a concrete GenAI architecture concept:

1. `01-citations-and-provenance/`
2. `02-hybrid/`
3. `03-graph-rag/`
4. `04-evaluation-and-orchestration/`

We use these to **tell the story** of how the system evolved, and to teach others.

---

## 4. Environment Setup

### 4.1 Prerequisites

* **Python**: use the version in `.python-version` (or any compatible Python 3.11+)
* **pip** (or `uv` / `pipx`, etc.)
* **Virtual environment** (recommended): `venv`, `conda`, or similar
* **OpenAI / other LLM provider account(s)** for embeddings + chat models
* Optionally: **Grok / X.ai**, or any other configured model provider

### 4.2 Clone the Repo

```bash
git clone https://github.com/ChicagoDro/AI-Portfolio.git
cd AI-Portfolio
```

### 4.3 Create and Activate a Virtual Env

```bash
# create venv
python -m venv .venv

# activate (macOS / Linux)
source .venv/bin/activate

# activate (Windows, PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 4.4 Install Dependencies

All Python dependencies are pinned in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you use `uv` or a different installer, it should still be able to consume `requirements.txt`.

### 4.5 Configure Environment Variables

Copy the example env file and fill in the secrets:

```bash
cp .env.example .env
```

Open `.env` and set the appropriate values. Typical entries include (exact names may vary; see `.env.example`):

* API keys for:

  * **Embeddings provider** (e.g., `OPENAI_API_KEY`)
  * **Chat/completion model provider(s)`** (OpenAI, Grok, etc.)
* Default model names for:

  * Embeddings
  * Chat / completion
* Optional tuning switches:

  * Temperature, max tokens
  * Whether to enable hybrid search / GraphRAG by default

> **Note:** Keep `.env` out of version control; it’s already in `.gitignore`.

---

## 5. Building the Indices

Before you can chat with the system, you need to ingest the data and build the indices.

From the repo root:

```bash
# run the ingestion + embedding pipeline
python src/RAG_Chatbot/ingest_embed_index.py
```

This will:

1. Load documents from `./data/Chitown_Custom_Choppers/`
2. Chunk and embed them
3. Write / refresh:

   * the **vector index** under `./indices/`
   * the **document metadata** under `./data/`
   * optionally the **knowledge graph** used for GraphRAG

If you add new documents or modify the corpus, re-run this script to rebuild the indices.

---

## 6. Running the Chatbot

Once your environment is configured and indices are built, you have two main ways to run the app.

### 6.1 Streamlit UI (recommended)

From the repo root:

```bash
streamlit run src/RAG_Chatbot/chitown_custom_choppers_chatbot.py
```

Then open the URL printed in the console (typically `http://localhost:8501`) and:

* Ask domain questions (“What’s the lead time for a custom build?”)
* Try more complex queries (“Which packages include both suspension and exhaust upgrades?”)
* Switch between / test hybrid and GraphRAG behaviors (depending on how you’ve wired the UI)


---

## 7. Learning Roadmap (Progressive Upgrades)

The `upgrades/RAG_Chatbot` directory documents your **learning journey**:

1. **Upgrade 1 – Citations & Provenance**

   * Add chunk IDs and source metadata
   * Display sources alongside answers
   * Demonstrates *grounded* RAG and trustworthiness

2. **Upgrade 2 – Hybrid Search (Dense + BM25)**

   * Introduce sparse retrieval (BM25)
   * Implement hybrid scoring / fusion
   * Show why hybrid beats pure vector search on many real tasks

3. **Upgrade 3 – GraphRAG**

   * Build an entity-level knowledge graph from the corpus
   * Use graph traversal + neighborhoods to retrieve structured context
   * Fuse graph and vector signals for richer, more coherent answers

4. **Upgrade 4 – Evaluation & Orchestration**

   * Add simple evaluation sets (Q/A pairs, metrics)
   * Introduce self-refinement, groundedness checks, and routing/orchestration patterns
   * Demonstrate how to treat RAG as a **system**, not a single chain

Each upgrade is intended to be a small, teachable module with:

* A short `README.md`
* Architecture diagrams / call graphs
* Design decisions and tradeoffs
* Key code snippets or diffs from the previous stage

---

## 8. For Employers & Collaborators

This repo is designed to show:

* **Engineering ability** – Python, vector stores, embeddings, Streamlit, etc.
* **Architecture thinking** – hybrid retrieval, GraphRAG, progressive upgrades
* **Documentation & teaching** – each step is explained, not just implemented
* **Growth mindset** – the roadmap continues with agents, evaluation, and deployment

If you’d like to discuss the design or extend it to your own domain, feel free to reach out.

---

## 9. About the Author

**Pete Tamisin** – Technical GTM Leader • AI & Data Engineering Architect • Builder & Teacher
Based in Chicago, IL.

* 20+ years designing data & AI platforms (Capital One, ex-Databricks, startups)
* Focused on **modern data platforms**, **RAG systems**, and **enterprise GenAI adoption**
* Passionate about **teaching** and helping teams ship real-world AI systems

📧 Email: `pete@tamisin.com`
🔗 LinkedIn: [peter-tamisin-50a3233a](https://www.linkedin.com/in/peter-tamisin-50a3233a/)

---

Feel free to fork, explore, and adapt this project for your own AI portfolio.

Welcome to **Chi-Town Custom Choppers** – where retrieval meets craftsmanship.
