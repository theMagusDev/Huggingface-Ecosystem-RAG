# Hugging Face Ecosystem RAG Assistant

An advanced **Retrieval-Augmented Generation (RAG)** assistant specialized in providing expert-level technical support for the Hugging Face ecosystem: `transformers`, `peft`, `accelerate`, `trl`, and `datasets`.

This project addresses the "hallucination" problem in LLMs regarding fast-moving library APIs by grounding responses in official documentation retrieved directly from GitHub.

## 🔗 Live Demo

**Try it now on Hugging Face Spaces:** [HF Ecosystem RAG Assistant](https://huggingface.co/spaces/yuriy-magus/huggingface_ecosystem_RAG)

*(No local setup required)*

---

## 🚀 Key Features

* **Semantic Library Routing**: Automatically classifies queries to specific libraries to narrow the search space and prevent cross-library noise.
* **Multi-Query Fusion**: Generates 3+ search variations for every user prompt to ensure high-recall retrieval even with vague technical terminology.
* **Transparent Reasoning**: Features a streaming interface that reveals the agent's internal "thought process" (Routing selection -> Query expansion -> Context extraction).
* **Automated Ingestion Pipeline**: Built-in ETL that clones repositories, parses Markdown headers, and builds/updates the ChromaDB vector index automatically.

---

## 🛠 Tech Stack

* **Orchestration**: LangChain (LCEL)
* **Vector Store**: ChromaDB
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local/CPU-optimized)
* **LLM**: Step-3.5-Flash (via OpenRouter)
* **UI Framework**: Gradio 5.0+ (Streaming support)
* **Data Handling**: GitPython (ETL), Git LFS (Database storage)

---

## 📁 Project Structure

```text
huggingface-ecosystem-RAG/
├── data/                 
│   ├── hf_vector_db/     # Persistent ChromaDB index (Stored via Git LFS)
│   └── temp_docs/        # Temporary storage for cloned HF repos
├── src/
│   ├── app.py            # Gradio Web UI & Streaming entry point
│   ├── rag_core.py       # Core RAG logic: Routing, Fusion, and Chains
│   ├── ingest.py         # ETL pipeline for documentation indexing
│   └── config.py         # Global configuration & Path management
├── .env                  # API Keys (OpenRouter, LangSmith)
└── requirements.txt      # Project dependencies

```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

Ensure you have [Git LFS](https://git-lfs.github.com/) installed to pull the pre-built vector database.

```bash
git clone https://github.com/your-username/huggingface-ecosystem-RAG.git
cd huggingface-ecosystem-RAG
git lfs pull
pip install -r requirements.txt

```

### 2. Configuration

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_openrouter_key
# Optional: Enable LangSmith for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_ls_key

```

### 3. Execution

```bash
python src/app.py

```

*Note: If the `data/` folder is empty, the system will automatically trigger the ingestion pipeline to build the index from scratch.*

---

## 🧠 Implementation Details

### Semantic Routing & Fallback

The agent employs a custom router to identify the target library meta-data. If the retrieval returns zero results within a specific library, a **Global Fallback** mechanism triggers a wider search across the entire ecosystem to find relevant context.
