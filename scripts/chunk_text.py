import argparse
import json
from pathlib import Path

from contract_rag.ingestion.chunking import (
    chunk_document,
    chunk_to_dict,
)
from contract_rag.models import (
    DocumentType,
    LegalDocument,
    PageSpan,
    SourceType,
)


def load_document(path: Path) -> LegalDocument:
    data = json.loads(path.read_text(encoding="utf-8"))

    # Construct directly so LegalDocument validates the stored hash,
    # normalized text, line offsets, and page spans.
    return LegalDocument(
        document_id=data["document_id"],
        title=data["title"],
        source_type=SourceType(data["source_type"]),
        original_text=data["original_text"],
        search_text=data["search_text"],
        content_sha256=data["content_sha256"],
        line_start_offsets=tuple(data["line_start_offsets"]),
        source_name=data.get("source_name"),
        document_type=DocumentType(data.get("document_type", "unknown")),
        page_spans=tuple(PageSpan(**page) for page in data.get("page_spans", [])),
        metadata=data.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create citation-preserving chunks from canonical JSON."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Canonical document JSON produced by ingest_text.py.",
    )
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks"),
    )
    arguments = parser.parse_args()

    document = load_document(arguments.file)
    chunks = chunk_document(
        document,
        max_chars=arguments.max_chars,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output / f"{document.document_id}.jsonl"

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "document_id": document.document_id,
                "chunk_count": len(chunks),
                "max_chars": arguments.max_chars,
                "output": str(output_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
