import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contract_rag.evaluation.cuad import (
    TARGET_CLAUSE_TYPES,
    answer_to_dict,
    iter_target_annotations,
    load_cuad,
)


def write_jsonl(
    output_path: Path,
    rows: Iterable[dict[str, Any]],
) -> int:
    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CUAD documents and evaluation queries.")
    parser.add_argument(
        "--input",
        default="data/cuad/test.json",
        help="Path to a CUAD JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cuad-test",
        help="Output directory.",
    )
    parser.add_argument(
        "--all-clauses",
        action="store_true",
        help="Export all CUAD clause types instead of the initial target set.",
    )

    arguments = parser.parse_args()

    input_path = Path(arguments.input)
    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    documents = load_cuad(input_path, validate_spans=True)

    document_rows = (
        {
            "document_id": document.document_id,
            "title": document.title,
            "context": document.context,
            "character_count": len(document.context),
        }
        for document in documents
    )

    document_count = write_jsonl(
        output_directory / "documents.jsonl",
        document_rows,
    )

    if arguments.all_clauses:
        selected_pairs = [
            (document, annotation) for document in documents for annotation in document.annotations
        ]
    else:
        selected_pairs = list(iter_target_annotations(documents))

    query_rows = []

    for document, annotation in selected_pairs:
        query_rows.append(
            {
                "query_id": annotation.query_id,
                "document_id": document.document_id,
                "title": document.title,
                "clause_type": annotation.clause_type,
                "question": annotation.question,
                "is_impossible": annotation.is_impossible,
                "answers": [answer_to_dict(answer) for answer in annotation.answers],
            }
        )

    query_count = write_jsonl(
        output_directory / "queries.jsonl",
        query_rows,
    )

    clause_counts = Counter(row["clause_type"] for row in query_rows)
    positive_count = sum(not row["is_impossible"] for row in query_rows)
    negative_count = query_count - positive_count

    summary = {
        "source": str(input_path.resolve()),
        "document_count": document_count,
        "query_count": query_count,
        "positive_query_count": positive_count,
        "negative_query_count": negative_count,
        "target_clause_types": (
            sorted(clause_counts) if arguments.all_clauses else sorted(TARGET_CLAUSE_TYPES)
        ),
        "queries_per_clause_type": dict(sorted(clause_counts.items())),
        "answer_spans_validated": True,
    }

    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCUAD preparation completed:\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
