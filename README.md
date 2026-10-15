# RAG Contract Analyzer

An evidence-grounded contract analysis project designed to retrieve relevant
clauses and produce answers with citations to the original document.

The initial focus is commercial-contract analysis, including liability and
termination clauses. A separate policy-analysis track uses terms of service
and privacy-policy documents.

## Current status

The dataset preparation pipeline is complete through batch chunking.
Embedding, vector indexing, and end-to-end RAG extraction are still pending.

### Implemented

- Python 3.12 project environment
- Local Ollama integration
- Docling PDF and DOCX parsing
- Canonical document representation
- SHA-256 content hashes
- Structured Markdown and JSON export
- Dataset adapters, including CUAD and ToS;DR
- Evidence-span validation against original text
- ToS;DR evidence-position recovery using unique exact matches
- Dataset integrity auditing
- Exact-text deduplication with source provenance
- Annotation and evaluation-reference remapping
- Section-aware chunking with exact citation positions
- Single-document and batch chunking scripts
- Automated unit tests

### Next steps

- Audit chunk lengths against an embedding model's token limit
- Benchmark local GPU embedding
- Generate embeddings
- Build a vector index
- Implement retrieval with document and source filters
- Evaluate retrieval against CUAD annotations

### Planned features

- Dense and sparse hybrid retrieval
- Reciprocal rank fusion
- Cross-encoder reranking
- Structured clause extraction using Ollama
- Answer-level citation and evidence validation
- FastAPI backend
- Streamlit interface

## How the system works

1. **Ingest documents:** extract text and source metadata.
2. **Standardize datasets:** convert source formats into a common document
   and annotation structure.
3. **Validate evidence:** check that quoted passages match their recorded
   character positions.
4. **Deduplicate documents:** consolidate identical text while preserving
   original source records and annotation links.
5. **Chunk documents:** split text into passages with exact citation positions.
6. **Embed and index:** convert passages into vectors for retrieval.
7. **Retrieve evidence:** find relevant passages for a question, filtered to
   the selected document or source scope.
8. **Generate an answer:** provide the question and retrieved passages to
   an LLM and return an answer with citations.
9. **Evaluate:** compare retrieved evidence and extracted answers with
   reference annotations.

Steps 6–9 are not yet implemented end to end.

Indexing documents does not train the LLM. Existing dataset annotations are
reference labels for evaluation and remain separate from retrieval text.

## Technology

| Component | Technology | Status |
|---|---|---|
| Language | Python 3.12 | In use |
| Document parsing | Docling and PyMuPDF | Available |
| Data preparation | Python and JSONL | Implemented |
| Testing and code quality | Pytest and Ruff | In use |
| GPU computation | PyTorch with CUDA | Locally verified |
| Embedding | Sentence Transformers | Planned |
| Initial embedding candidate | `BAAI/bge-small-en-v1.5` | Benchmark pending |
| Vector database | Qdrant | Planned |
| Local LLM | Ollama with Qwen | Integration available; extraction pending |
| Backend | FastAPI | Planned |
| Interface | Streamlit | Planned |

## Setup

Create and activate a Python 3.12 virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the local project:

```powershell
python -m pip install -e .
```

Run subsequent commands from the repository root.

### GPU setup

The following combination was verified locally on Windows with an NVIDIA
RTX 4060 Laptop GPU:

- Python 3.12
- PyTorch `2.9.1+cu128`
- torchvision `0.24.1+cu128`

Before changing an existing environment, inspect its packages:

```powershell
python -m pip show torch torchvision torchaudio
python -m pip check
```

For a compatible environment requiring this specific GPU build:

```powershell
python -m pip install "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip check
```

These commands install packages into the active environment. They do not
automatically update the project's dependency configuration.

## Source datasets

### CUAD

[Contract Understanding Atticus Dataset](https://www.atticusprojectai.org/cuad)
contains commercial contracts with clause questions and annotated evidence.

Our imported snapshot contains:

- 510 contracts
- 20,910 clause-question annotations
- 6,702 positive annotations with verified evidence
- 14,208 negative examples

A negative example indicates that the source dataset provides no answer
span for that contract and question. Retain these examples when evaluating
whether the system avoids unsupported answers.

### ToS;DR

[Terms of Service; Didn't Read](https://tosdr.org/) provides annotations
about online services' terms and privacy policies.

The local run uses the
[ToS;DR raw export on Zenodo](https://zenodo.org/records/15012282):

- `documents.csv`: document text and metadata
- `points.csv`: annotations, quotes, and source positions

Our imported snapshot contains:

- 16,603 documents
- 27,365 annotations
- 13,965 annotations with verified evidence positions
- 13,400 annotations without verified evidence

ToS;DR extends the corpus into policy analysis. Its annotations represent a
different task from CUAD and should be evaluated separately.

## Supported dataset adapters

`scripts/prepare_dataset.py` converts source data into the canonical
`LegalDocument` format.

| Dataset argument | Expected input | Output coverage |
|---|---|---|
| `cuad` | CUAD SQuAD-style JSON | Contract text, clause questions, and validated answer spans |
| `tosdr` | `documents.csv` and optional `points.csv` | Document text, metadata, point labels, and checked quote positions |
| `open_terms` | Open Terms Archive versions directory | Markdown versions with path-based provenance |
| `contractnli` | ContractNLI JSON | Contract text, hypotheses, labels, and evidence spans |
| `maud` | SQuAD-style MAUD JSON export | Agreement text and answer annotations |
| `mcc` | Directory of TXT or Markdown files | Document text and relative-path provenance |
| `opp115` | Directory of TXT or Markdown files | Policy text and relative-path provenance |

Only CUAD and ToS;DR are included in the completed combined-corpus run.

MAUD requires a SQuAD-style export. MCC and OPP-115 require extracted text
or Markdown. Native OPP-115 annotations are not supported by this adapter.

Each source export contains:

| File | Purpose |
|---|---|
| `documents.jsonl` | Original document text and metadata |
| `annotations.jsonl` | Existing labels and supporting evidence |
| `evaluation.jsonl` | Evaluation cases produced by the adapter |
| `source_manifest.json` | Input paths, counts, and source notes |

The ToS;DR adapter currently exports annotations without generating evaluation
questions, so its `evaluation.jsonl` is empty.

## Preparing the combined corpus

### 1. Download and verify source files

Store downloaded files at:

```text
data/raw/cuad/CUAD_v1.json
data/raw/tosdr/documents.csv
data/raw/tosdr/points.csv
```

Verify downloads against the checksums published for the chosen source
version. Raw datasets are not included in this repository.

### 2. Export CUAD and ToS;DR

```powershell
python scripts/prepare_dataset.py --dataset cuad --input data/raw/cuad/CUAD_v1.json --output data/processed/datasets/cuad_v2
python scripts/prepare_dataset.py --dataset tosdr --input data/raw/tosdr/documents.csv --points data/raw/tosdr/points.csv --output data/processed/datasets/tosdr_v2
```

The `_v2` paths identify the corrected exports used in this local workflow.
They are not upstream dataset version numbers.

The adapters convert existing annotations; they do not use an LLM to create
new labels.

### 3. Understand ToS;DR evidence validation

The ToS;DR adapter:

1. Reads quoted evidence from `quote_text`.
2. Checks whether the quote matches the supplied offsets using an exclusive
   end position.
3. Checks the inclusive-end interpretation if needed.
4. Searches for a unique exact occurrence if the supplied positions fail.
5. Leaves ambiguous or unmatched evidence unverified.

All output spans use an inclusive start and exclusive end:

```python
original_text[start_char:end_char]
```

Original text is preserved so character positions remain meaningful.
No fuzzy matching is used to mark evidence as verified.

Observed results:

| Matching method | Annotation count |
|---|---:|
| Supplied exclusive-end offsets | 13,061 |
| Unique exact occurrence recovered | 904 |
| Ambiguous occurrence | 57 |
| Quote absent from linked text | 13,342 |
| Missing quote | 1 |

The 51.03% verified-evidence rate measures text alignment, not model accuracy
or independent legal correctness.

### 4. Audit the datasets

In `scripts/audit_datasets.py`, configure:

```python
DATASETS = {
    "cuad": ROOT / "data/processed/datasets/cuad_v2",
    "tosdr": ROOT / "data/processed/datasets/tosdr_v2",
}
```

Run:

```powershell
python scripts/audit_datasets.py
```

The audit checks:

- JSONL parsing
- Document content hashes
- Duplicate document IDs
- Empty documents
- Annotation-to-document references
- Evidence text and character positions
- Negative examples containing unexpected spans
- Exact duplicate document texts

Resolve integrity errors before merging.

### 5. Merge and deduplicate

```powershell
python scripts/merge_datasets.py
```

The script reads the two corrected exports and writes:

```text
data/processed/datasets/combined_v1/
```

It groups documents by SHA-256 of their unchanged original text.
Annotation and evaluation references are remapped to canonical document IDs.

| Output | Purpose |
|---|---|
| `documents.jsonl` | One document per distinct exact text |
| `document_aliases.jsonl` | Original-to-canonical document ID mapping |
| `annotations.jsonl` | All annotations with updated document links |
| `annotations_verified.jsonl` | Verified evidence annotations plus CUAD negative examples |
| `evaluation.jsonl` | Existing evaluation cases with updated document links |
| `source_manifest.json` | Counts, input paths, and original source manifests |

The first encountered document supplies the representative title and
document type. All original source records remain available in
`metadata.source_records`.

Use `metadata.source_datasets` for source membership because a deduplicated
document can belong to multiple sources.

Annotations are retained individually even when their document text is
identical. Exact-text deduplication does not detect near-duplicates.

The script refuses to overwrite an existing output directory. Choose a
new `OUTPUT` path in the script for another run.

## Chunking

The shared implementation is:

```text
src/contract_rag/ingestion/chunking.py
```

### Single-document chunking

`scripts/chunk_text.py` processes one canonical document JSON file:

```powershell
python scripts/chunk_text.py --file "path/to/document.json"
```

Replace the example path with an actual document JSON file. This command
does not read a multi-record dataset JSONL file.

### Batch chunking

`scripts/chunk_dataset.py` processes the merged JSONL corpus using the same
chunking implementation:

```powershell
python scripts/chunk_dataset.py --input data/processed/datasets/combined_v1/documents.jsonl --output data/processed/chunks/combined_v1 --max-chars 1800
```

The output directory must not already exist.

Current behavior:

- Maximum 1,800 characters per chunk
- Section-aware splitting
- Preference for paragraph, line, or whitespace boundaries
- No overlap between chunks
- Exact document IDs and character positions
- Complete source-text coverage
- Citation-position validation during export
- Source metadata retained for retrieval filtering

Outputs:

| File | Purpose |
|---|---|
| `chunks.jsonl` | Chunk text, IDs, locations, and source metadata |
| `chunk_manifest.json` | Settings, counts, and validation results |

Annotation labels and reference answers are not included in chunk text.

Character limits are not token limits. Before embedding, check every chunk
against the selected model's tokenizer and split oversized chunks without
losing citation positions.

Whitespace-only chunks should be skipped during embedding.

## Validated local results

### Source data

| Measure | CUAD | ToS;DR |
|---|---:|---:|
| Documents before deduplication | 510 | 16,603 |
| Annotations | 20,910 | 27,365 |
| Positive annotations with verified evidence | 6,702 | 13,965 |
| Negative examples | 14,208 | 0 |
| Annotations without verified evidence | 0 | 13,400 |

### Combined document audit

- 17,113 input document records
- 12,713 distinct exact texts
- 105 duplicate-text groups
- 4,400 extra duplicate records
- No invalid evidence spans detected
- No orphan annotations detected
- No duplicate document IDs detected
- No incorrect content hashes detected
- No empty documents detected

### Batch chunking

- 12,713 documents processed
- 188,455 chunks produced
- Average chunk length: 1,508.97 characters
- 31 whitespace-only chunks
- 188,424 nonblank candidate chunks before token-limit adjustments
- All citation positions validated
- Complete text coverage confirmed

These results describe data integrity. Retrieval quality and answer accuracy
have not yet been measured.

## Retrieval and evaluation design

- Retrieve from the selected document when answering questions about one
  specific contract.
- Retain source filters for commercial-contract and policy-analysis workflows.
- Keep reference annotations separate from indexed text.
- Evaluate CUAD and ToS;DR separately rather than relying on one combined score.
- Retain CUAD negative examples to test unsupported-answer behavior.
- Exclude unverified ToS;DR evidence from the verified evaluation subset.
- Keep identical documents together when creating evaluation splits.
- Assess near-duplicates before establishing final evaluation splits.

Evaluation-split construction, retrieval metrics, and answer-level evaluation
remain pending.

## Development checks

With development tools installed:

```powershell
python -m ruff check src scripts tests
python -m pytest -q
```

Run adapter tests only:

```powershell
python -m pytest -q tests/test_dataset_adapters.py
```

Format Python files:

```powershell
python -m ruff format src scripts tests
```

## Data storage and version control

The repository ignores:

- Raw datasets and processed exports
- Uploaded contracts
- Chunks, embeddings, and vector indexes
- Downloaded model files
- Virtual environments and caches
- Local databases and logs
- Environment secrets

Commit source code, tests, documentation, and dependency configuration.

Check for tracked files that match ignore rules:

```powershell
git ls-files -ci --exclude-standard
```

An empty result means no currently tracked files match those rules.
Adding a file to `.gitignore` does not remove it from existing Git history.

## Dataset rights and limitations

Dataset access does not automatically grant permission to redistribute
underlying contracts or policies.

Review each source dataset's license and underlying-document terms before
redistribution or commercial use. The preparation CLI accepts `--license-note`
for recording source-specific notes.

Other limitations:

- Source annotations may be incomplete or refer to different document versions.
- Exact evidence alignment does not prove legal correctness.
- Exact-text deduplication does not remove near-duplicates.
- A clause can span multiple non-overlapping chunks.
- The initial embedding candidate targets English; language coverage must
  be assessed before broader use.
- The system is a development project and has not been validated for legal
  decision-making.