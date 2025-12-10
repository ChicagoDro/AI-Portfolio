
# Chi-Town Custom Choppers RAG Capstone  
### *A Progressive Learning Repository for Building Real-World GenAI Retrieval Systems*

Welcome!  
This repository is intentionally structured as **both**:

1. A **production-quality RAG project** you can run today  
2. A **progressive, educational curriculum** showing the step-by-step path toward becoming an **AI Solution Architect**

It is designed for employers, collaborators, and learners who want to see:

- Clean, well-designed GenAI systems  
- Real retrieval architectures you’d build to customize responses from LLMs like ChatGpt or Grok  
- A documented roadmap showing *how* each capability was added  
- A portfolio demonstrating deep mastery, not surface-level demos  

---

# Project Purpose

This repository is your personal **GenAI mastery journey**.  

It begins with a simple RAG system, then evolves through increasingly advanced architectural upgrades:

- vector retrieval  
- hybrid search  
- citations and provenance  
- query rewriting  
- graph-based retrieval (GraphRAG)  
- validation and self-refinement  
- orchestration  
- evaluation pipelines  

Each step is documented so anyone following your footsteps can learn with you — and employers can clearly see your growth and technical depth.

---

# Repository Structure

```text
RAG-Capstone/
├── README.md                    # ← You are here
├── requirements.txt
├── .env.example
├── src/
│   └── RAG_Chatbot/
│       ├── ingest_embed_index.py
│       ├── chitown_custom_choppers_chatbot.py
│       ├── graph_kg_builder.py
│       ├── graph_retrieval.py
│       └── config.py
├── data/
│   ├── document-metadata.json
│   └── Chitown_Custom_Choppers/            # (Optional; usually .gitignored)
├── indices/                     # Vector + graph indices (generated)
├── notebooks/                   # Experimentation + learning scratchpad
└── upgrades/                    # ← Progressive learning modules
    ├── 01-basic-rag/
    ├── 02-citations-and-hybrid/
    ├── 03-graph-rag/
    └── 04-evaluation-and-orchestration/
````

### ✔ `src/`

The **current, stable implementation** of the Chi-Town Custom Choppers RAG system.

### ✔ `upgrades/`

A **progressive curriculum**.
Each folder contains:

* A dedicated `README.md` explaining the concept
* Architecture diagrams
* Design decisions
* Key code snippets or diffs
* Why this upgrade matters in real-world GenAI systems

This is where you demonstrate *architect-level thinking*.

### ✔ `notebooks/`

Your scratch environment for exploration and prototyping.
Shows your research process — something employers value.

---

# Running the Main Application

After installing dependencies:

```bash
python ingest_embed_index.py
streamlit run cozy_corner_chatbot.py
```

The system supports:

* OpenAI embeddings
* OpenAI or Grok LLMs
* Metadata filtering
* Classification + retrieval fusion
* Future: GraphRAG, hybrid search, validation, and more

---

# Learning Roadmap (Progressive)

This repository is organized into a learning journey:

### **Upgrade 1: Basic RAG**

Builds a foundational vector→retriever→LLM pipeline.

### **Upgrade 2: Citations + Chunk Scoring + Hybrid Search**

Adds production features: provenance, BM25, fusion.

### **Upgrade 3: GraphRAG**

Adds entity extraction, knowledge graph construction, graph-based retrieval, and vector+graph fusion.

### **Upgrade 4: Evaluation + Orchestration**

Adds:

* groundedness checks
* self-refinement
* orchestrated workflows
* evaluation sets

Each upgrade teaches a real GenAI system design concept used in industry.

---

# 👥 For Employers

This repo shows:

* Engineering ability (Python, LangChain, embeddings, FAISS, Streamlit)
* Architecture thinking (pipelines, modularity, hybrid retrieval)
* Deepening capability (from basic RAG → GraphRAG → evaluation)
* Growth mindset and documentation skills
* Ability to communicate complex systems clearly

You can explore the `upgrades/` directory to see the progression from novice → architect-level designs.

---

# For Learners

Each upgrade folder is a **mini-course**.
You can follow along step by step to learn:

* how RAG really works
* how to improve retrieval quality
* how to build a knowledge graph from text
* how to fuse graph and vector search
* how to write production-ready GenAI pipelines

This repository is designed to be forked and extended.

---

# Future Goals

This repository will continue to evolve into:

* Multimodal RAG
* Agentic workflows
* Evaluation frameworks
* Deployment examples
* Model routing & advanced pipelines

---

# Closing

This project is both:

* a **portfolio piece** showing real, deep GenAI capability, and
* a **learning resource** for anyone who wants to follow the same journey.

Feel free to explore, fork, study, and build on it.

Welcome to the Chi-Town Custom Choppers — where retrieval meets craftsmanship.

```
```
