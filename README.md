Akanksha AI: Personal Second Brain and Research Assistant
=========================================================

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Vector DB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![Framework](https://img.shields.io/badge/FastAPI%20%2F%20Starlette-REST%20%26%20Web-teal.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A local Retrieval-Augmented Generation (RAG) system to organize, index, and query personal documents, research papers, study notes, and code.

---

What is this project?
---------------------

Akanksha AI is an open-source, private AI Second Brain and Research Assistant built with Python and ChromaDB.

As a developer, student, or researcher, you accumulate dozens of PDF papers, Markdown notes, Jupyter notebooks, and code files across your computer. Finding specific formulas, explanations, or algorithms across all these files usually requires tedious manual searching.

This project solves that problem by turning your local files into an intelligent, interactive personal knowledge base that you can search and chat with on your own computer.

---

What does it do?
----------------

1. Ingests and parses your files
It reads PDFs, text documents, markdown notes, source code, and Jupyter notebooks, dividing them into overlapping semantic passages while preserving page numbers and source filenames.

2. Builds a local vector index
It converts your document passages into vector embeddings using ChromaDB and builds a keyword index, storing everything securely on your local machine.

3. Performs hybrid semantic search
When you search or ask a question, it uses Reciprocal Rank Fusion (RRF) to combine dense vector similarity with exact keyword matching, ensuring that both broad concepts and exact technical terms are found accurately.

4. Delivers grounded answers with exact citations
Instead of making up answers, it retrieves the exact passages relevant to your question, synthesizes a clear answer, and points you directly to the source filename, page number, and similarity score.

5. Offers two user interfaces
- Web Studio: A modern, browser-based chat dashboard with drag-and-drop document uploads and clickable citation cards.
- Terminal CLI: A command-line tool to quickly index files, search snippets, or chat directly from PowerShell or Command Prompt.

6. Works 100% offline
It runs completely locally without requiring an internet connection or paid API keys. If you want full generative answers, you can optionally connect free Google Gemini, OpenAI, or local Ollama models in the settings.

---

Features
--------

- Multi-format ingestion: Supports PDF (with page-level tracking), Markdown, TXT, Python, JavaScript, JSON, CSV, and Jupyter Notebooks (.ipynb).
- Source citations: Every response cites source filenames, page numbers, and similarity confidence scores.
- Hybrid search: Combines semantic embeddings (all-MiniLM-L6-v2) with exact keyword matching.
- Offline first: Built-in local extractive synthesizer runs with zero configuration and no API keys.
- Pluggable LLM backends: Easily switch between Offline, Google Gemini, OpenAI, or Ollama in settings.
- Web Studio: Single-page application with drag-and-drop upload, chat, and citation inspection.
- Terminal CLI: Command-line interface for scripting, ingestion, and terminal chat.

---

Quickstart
----------

1. Install Dependencies

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/akanksha-bharti579/akanksha.git
cd akanksha
pip install -e .
```

2. Ingest a Document

Ingest sample research notes:

```bash
python -m akanksha ingest examples/deep_learning_and_rag_notes.md
```

Or ingest any external document or directory:

```bash
python -m akanksha ingest "path/to/document.pdf"
```

3. Ask a Question

Ask questions grounded directly in your indexed documents:

```bash
python -m akanksha ask "What is the Scaled Dot-Product Attention formula?"
```

4. Launch the Web Interface

Start the local web application:

```bash
python -m akanksha web
```

Open http://127.0.0.1:8000 in your web browser.

---

Command Line Interface (CLI)
----------------------------

| Command | Description | Example |
|---|---|---|
| python -m akanksha ingest <path> | Ingest a file or directory | python -m akanksha ingest ./notes |
| python -m akanksha list | List all indexed documents | python -m akanksha list |
| python -m akanksha ask "<query>" | Ask a question with citations | python -m akanksha ask "Explain RAG" |
| python -m akanksha search "<query>" | Semantic search across chunks | python -m akanksha search "gradient boosting" |
| python -m akanksha summarize "<title>" | Summarize an indexed document | python -m akanksha summarize notes.md |
| python -m akanksha chat | Start interactive terminal chat | python -m akanksha chat |
| python -m akanksha web | Start the web application | python -m akanksha web --port 8000 |
| python -m akanksha clear | Clear all indexed vector data | python -m akanksha clear |

---

Python SDK Usage
----------------

You can use the SecondBrain engine directly inside Python scripts:

```python
from akanksha import SecondBrain

brain = SecondBrain()
brain.ingest_file("machine_learning_notes.pdf")

result = brain.query("How does Gradient Boosting differ from Random Forest?")

print("Answer:\n", result.answer)
print("\nSources:")
for src in result.sources:
    print(f"- {src.title} (Page {src.page_number}) [Score: {src.similarity_score:.2f}]")
```

---

LLM Providers
-------------

The engine runs offline by default. You can configure other providers in the web interface or via environment variables:

| Provider | Configuration | Description |
|---|---|---|
| Offline Extractive (Default) | No setup required | Runs locally, zero cost, instant response |
| Google Gemini | GEMINI_API_KEY="AIza..." | Fast generative answers via Gemini 1.5 Flash |
| OpenAI / Groq | OPENAI_API_KEY="sk-..." | GPT-4o-mini or OpenAI-compatible endpoints |
| Local Ollama | OLLAMA_BASE_URL="http://localhost:11434" | Private local models such as Llama 3 or Mistral |

---

Project Structure
-----------------

```
akanksha/
├── akanksha/                   (Main Python package)
│   ├── core/                   (RAG Engine modules)
│   │   ├── config.py           (Configuration and settings)
│   │   ├── document_loader.py  (File parsers for PDF, MD, Code, Notebooks)
│   │   ├── text_splitter.py    (Recursive chunking logic)
│   │   ├── embeddings.py       (Embedder connectors)
│   │   ├── vector_store.py     (ChromaDB and hybrid search logic)
│   │   ├── llm_provider.py     (LLM backends)
│   │   └── rag_engine.py       (Main SecondBrain orchestrator)
│   ├── cli.py                  (Terminal interface)
│   ├── web/                    (Web Studio application)
│   │   ├── server.py           (Starlette REST backend)
│   │   └── static/             (Frontend HTML, CSS, JavaScript)
│   ├── __init__.py             (Public API exports)
│   └── __main__.py             (CLI entrypoint)
├── examples/                   (Sample documents and quickstart script)
│   ├── deep_learning_and_rag_notes.md
│   └── quickstart.py
├── tests/                      (Automated unit tests)
│   └── test_rag.py
├── pyproject.toml              (Packaging configuration)
├── requirements.txt            (Python dependencies)
├── .gitignore                  (Git ignore rules)
└── README.md                   (Documentation)
```

---

Running Tests
-------------

To run the unit test suite:

```bash
python -m pytest tests/ -v
```

---

License
-------

Distributed under the MIT License. Created by Akanksha Bharti.
