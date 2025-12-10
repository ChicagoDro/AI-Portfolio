#  Hi, I’m **Pete Tamisin**
### **Director of Solutions & Data Strategy • AI & Data Engineering Architect • Builder & Teacher**

I’m a Director-level technical leader with 20+ years of experience designing **modern data platforms**, driving **enterprise AI adoption**, and scaling high-growth products. My work bridges the worlds of **architecture**, **customer success**, and **business strategy** to deliver real-world impact across Fortune 500 enterprises and fast-moving startups.  

---

##  What I Do
- **Modern Data & AI Architecture** — Databricks, Apache Spark, Snowflake, RAG systems, LangChain  
- **Solution Architecture & Field Enablement** — global POV frameworks, enterprise onboarding, pre/post-sales  
- **Technical Product Scaling** — taking a startup MVP to an enterprise-grade platform trusted by global brands  
- **Hands-on Leadership** — coaching engineers, shaping product strategy, driving customer adoption

---

##  What I’m Known For
- Turning complex AI/data problems into clear, actionable designs  
- High-energy teaching and technical enablement (45+ presenters trained for major events)  
- Leading every closed enterprise deal at Sync Computing as the SA on 100% of contracts  
- Supporting executive teams through acquisition diligence and growth strategy  
- Delivering millions in customer savings and uncovering critical product edge cases

---

##  Why Students & Teams Work With Me
I believe in teaching the **why**, not just the **how**.  
I help teams build confidence, understand architecture deeply, and design systems that **last**.

Whether you're learning AI, exploring data engineering, or building cloud-native systems, my goal is simple:

**Empower you to build, ship, and grow.**

---

## 🛠️ Tech Stack I Work With
**Platforms:** Databricks • Apache Spark • Snowflake • Kafka  
**AI/LLM Tools:** LangChain • RAG architectures • Vector DBs  
**Languages:** Python • SQL • Scala • JavaScript • R • Java  
**DevOps & Cloud:** AWS • Azure • Airflow • Docker • Terraform • GitHub Actions  

---

## Let’s Connect
**Email:** pete@tamisin.com

**LinkedIn:** [https://www.linkedin.com/in/peter-tamisin-50a3233a/](https://www.linkedin.com/in/peter-tamisin-50a3233a/)  

**Location:** Chicago, IL  

Always open to collaborating, mentoring, or helping teams level up their data & AI capabilities.



# Chi-Town Custom Choppers RAG Example  
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
├── .env.example (.env .gitnored)
├── src/
│   └── RAG_Chatbot/
│       ├── ingest_embed_index.py
│       ├── chitown_custom_choppers_chatbot.py
│       ├── graph_kg_builder.py
│       ├── graph_retrieval.py
│       └── config.py
├── data/
│   ├── document-metadata.json
│   └── Chitown_Custom_Choppers/            # (LLM generated company data files)
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
