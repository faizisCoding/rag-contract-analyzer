import argparse
import json
import sys
from pathlib import Path

from contract_rag.ingestion.text import (
    ingest_pasted_text,
    ingest_txt,
)
from contract_rag.models import legal_document_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Ingest a TXT file or piped text into the canonical legal-document format.")
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--file",
        help="Path to a UTF-8 .txt legal document.",
    )

    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read pasted text from standard input.",
    )

    parser.add_argument(
        "--title",
        help="Optional document title.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/ingested",
        help="Directory for the canonical JSON output.",
    )

    arguments = parser.parse_args()

    if arguments.file:
        document = ingest_txt(
            arguments.file,
            title=arguments.title,
        )
    else:
        document = ingest_pasted_text(
            sys.stdin.read(),
            title=(arguments.title or "Pasted legal document"),
        )

    output_directory = Path(arguments.output).resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / f"{document.document_id}.json"

    output_path.write_text(
        json.dumps(
            legal_document_to_dict(document),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "document_id": document.document_id,
        "title": document.title,
        "source_type": document.source_type.value,
        "character_count": document.character_count,
        "line_count": document.line_count,
        "output": str(output_path),
    }

    print("\nText ingestion completed:\n")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
