"""Batch-chunk a merged corpus while validating citation positions."""

import argparse
import json
from pathlib import Path

from contract_rag.ingestion.chunking import (
    CHUNKER_VERSION,
    chunk_document,
    chunk_to_dict,
)
from contract_rag.models import (
    DocumentType,
    LegalDocument,
    PageSpan,
    SourceType,
)


def load_document(data):
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
        page_spans=tuple(
            PageSpan(**page) for page in data.get("page_spans", [])
        ),
        metadata=data.get("metadata", {}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    if args.max_chars < 2:
        parser.error("--max-chars must be at least 2")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.output.exists():
        parser.error("Output folder already exists; choose a new folder.")

    args.output.mkdir(parents=True)
    partial = args.output / "chunks.partial.jsonl"
    final = args.output / "chunks.jsonl"

    document_count = 0
    chunk_count = 0
    whitespace_chunks = 0
    total_characters = 0
    seen_documents = set()

    with (
        args.input.open(encoding="utf-8") as reader,
        partial.open("w", encoding="utf-8", newline="\n") as writer,
    ):
        for line in reader:
            if not line.strip():
                continue

            document = load_document(json.loads(line))
            if document.document_id in seen_documents:
                raise ValueError(
                    f"Duplicate document ID: {document.document_id}"
                )
            seen_documents.add(document.document_id)

            chunks = chunk_document(
                document, max_chars=args.max_chars
            )
            cursor = 0

            for chunk in chunks:
                location = chunk.location
                if (
                    location.document_id != document.document_id
                    or location.start_char != cursor
                    or not cursor < location.end_char
                    <= len(document.original_text)
                    or chunk.text
                    != document.original_text[
                        location.start_char:location.end_char
                    ]
                    or len(chunk.text) > args.max_chars
                ):
                    raise ValueError(
                        f"Invalid chunk: {chunk.chunk_id}"
                    )

                cursor = location.end_char
                record = chunk_to_dict(chunk)

                # Small metadata fields for later retrieval filtering.
                # Full provenance remains in documents.jsonl.
                record["metadata"] = {
                    "source_datasets": document.metadata.get(
                        "source_datasets", []
                    ),
                    "title": document.title,
                    "document_type": document.document_type.value,
                }

                writer.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                chunk_count += 1
                total_characters += len(chunk.text)
                whitespace_chunks += int(not chunk.text.strip())

            if cursor != len(document.original_text):
                raise ValueError(
                    f"Incomplete coverage: {document.document_id}"
                )

            document_count += 1
            if document_count % 500 == 0:
                print(
                    f"Processed {document_count:,} documents; "
                    f"{chunk_count:,} chunks",
                    flush=True,
                )

    if document_count == 0:
        raise ValueError("Input contains no documents.")

    partial.rename(final)

    manifest = {
        "input": str(args.input.resolve()),
        "chunker_version": CHUNKER_VERSION,
        "max_chars": args.max_chars,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "whitespace_only_chunks": whitespace_chunks,
        "average_chunk_characters": round(
            total_characters / chunk_count, 2
        ),
        "citation_positions_validated": True,
        "complete_text_coverage": True,
    }

    (args.output / "chunk_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()