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
## Preparing external datasets

`scripts/prepare_dataset.py` normalizes source documents into the same canonical
`LegalDocument` model used by the analyzer. It writes separate `documents.jsonl`,
`annotations.jsonl`, `evaluation.jsonl`, and `source_manifest.json` files. Raw
downloads are not copied into the repository.

Supported adapters:

| `--dataset` | Input | What the adapter preserves |
|---|---|---|
| `cuad` | CUAD SQuAD-style JSON | Full contract text and validated clause answer offsets |
| `tosdr` | ToS;DR `documents.csv`; optional `--points points.csv` | Full text, service URL, point labels, and quote offsets (inclusive source end converted to exclusive) |
| `open_terms` | Extracted Open Terms Archive versions directory | Each Markdown version as a distinct document with service/type/version path metadata |
| `contractnli` | ContractNLI JSON | Contract text, NLI labels, hypotheses, and evidence spans |
| `maud` | SQuAD-style MAUD JSON export | Agreement text and validated answer offsets |
| `mcc` | Extracted directory of TXT/Markdown files | Agreement text and relative path provenance |
| `opp115` | Extracted directory of TXT/Markdown policy files | Policy text and relative path provenance |

Examples in PowerShell:

```powershell
python scripts/prepare_dataset.py --dataset cuad --input data/raw/cuad/test.json
python scripts/prepare_dataset.py --dataset tosdr --input data/raw/tosdr/documents.csv --points data/raw/tosdr/points.csv
python scripts/prepare_dataset.py --dataset open_terms --input data/raw/open-terms-archive/contrib-versions
python scripts/prepare_dataset.py --dataset contractnli --input data/raw/contractnli/train.json
```

MAUD must be supplied as a SQuAD-style JSON export; MCC and OPP-115 must be
extracted to text/Markdown files first. OPP-115's annotation package is not
flattened by this first adapter version, so its policies are currently corpus
documents only. Review each source's license and underlying-document terms;
record source-specific notes with `--license-note` before sharing prepared data.
For example, MCC is non-commercial and share-alike, and CUAD's license does not
settle rights in the underlying contracts.
