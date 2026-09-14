# RAG Contract Analyzer

An evidence-grounded legal contract analysis system built to retrieve and
extract liability and termination clauses from commercial agreements.

## Current status

Implemented:

- Python 3.12 project environment
- Local Ollama integration
- Docling PDF and DOCX parsing
- SHA-256 document identification
- Structured Markdown and JSON export
- CUAD dataset loading and validation
- Offset validation for annotated evidence
- CUAD evaluation-data preparation
- Automated unit tests

Planned:

- Offset-preserving legal chunking
- Dense and sparse hybrid retrieval
- Reciprocal rank fusion
- Cross-encoder reranking
- Structured clause extraction using Ollama
- Citation and evidence validation
- Retrieval evaluation
- FastAPI and Streamlit interfaces

## Technology

- Python 3.12
- Docling and PyMuPDF
- Qdrant and FastEmbed
- Sentence Transformers
- Ollama with Qwen
- FastAPI
- Streamlit
- Pytest and Ruff

## Setup

Create and activate a Python 3.12 environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1