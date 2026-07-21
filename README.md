# Diabetes in Pregnancy - Clinical Assistant (Offline RAG)

A local, offline RAG (Retrieval-Augmented Generation) application that provides guideline-based clinical recommendations from NICE NG3: *"Diabetes in pregnancy: management from preconception to the postnatal period."*

Healthcare professionals can query the tool and receive evidence-based recommendations without needing to manually reference the guideline document.

## Tech Stack

- **LLM**: Ollama (llama3.1:8b) - runs fully offline
- **Embeddings**: nomic-embed-text (via Ollama)
- **Vector Store**: FAISS (local, no cloud)
- **Framework**: LangChain
- **Frontend**: Streamlit
- **PDF Parsing**: pypdf

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- [Ollama](https://ollama.ai/) installed and running

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajeshskollam/ai-rag-offline-demo.git
   cd ai-rag-offline-demo
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml --prefix ./.env
   conda activate ./.env
   ```

3. **Pull the required Ollama models:**
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

4. **Place the guideline PDF** in the `documents/` folder (not included in repo due to copyright).

5. **Run the application:**
   ```bash
   python -m streamlit run frontend_interface.py
   ```

The app will automatically index the PDF on first run and save the FAISS index locally for subsequent launches.

## Example Query

> How to manage 24 weeks Primi Gravida with 24gm OGTT value is 174?

The assistant provides structured recommendations covering diagnosis, dietary advice, blood glucose targets, pharmacological management, monitoring schedule, and referral actions.

## Project Structure

```
├── backend_processor.py    # RAG pipeline (embeddings, retrieval, LLM)
├── frontend_interface.py   # Streamlit chat interface
├── environment.yml         # Conda environment specification
├── documents/              # Place guideline PDF here
├── faiss_index/            # Auto-generated vector index (gitignored)
└── .env/                   # Conda environment (gitignored)
```
