"""Audit exported datasets without changing their files."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cuad": ROOT / "data/processed/datasets/cuad_v2",
    "tosdr": ROOT / "data/processed/datasets/tosdr_v2",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}, line {line_number}"
                    ) from exc


def main():
    hashes = Counter()
    all_document_ids = set()
    totals = Counter()

    for dataset, folder in DATASETS.items():
        counts = Counter()
        annotations_by_document = defaultdict(list)

        for annotation in read_jsonl(folder / "annotations.jsonl"):
            annotations_by_document[annotation["document_id"]].append(annotation)
            counts["annotations"] += 1

        for document in read_jsonl(folder / "documents.jsonl"):
            counts["documents"] += 1
            document_id = document["document_id"]
            text = document["original_text"]

            if document_id in all_document_ids:
                counts["duplicate_document_ids"] += 1
            all_document_ids.add(document_id)

            if not text.strip():
                counts["empty_documents"] += 1

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            hashes[digest] += 1

            if digest != document.get("content_sha256"):
                counts["incorrect_document_hashes"] += 1

            annotations = annotations_by_document.pop(document_id, [])
            for annotation in annotations:
                spans = annotation.get("evidence_spans") or []
                valid_spans = True

                for span in spans:
                    start = span.get("start_char")
                    end = span.get("end_char")
                    valid = (
                        type(start) is int
                        and type(end) is int
                        and 0 <= start < end <= len(text)
                        and text[start:end] == span.get("text")
                    )
                    if not valid:
                        counts["invalid_evidence_spans"] += 1
                        valid_spans = False

                if annotation.get("is_impossible") is True:
                    counts["negative_examples"] += 1
                    if spans:
                        counts["negative_examples_with_spans"] += 1
                elif spans and valid_spans:
                    if annotation.get("offsets_validated") is True:
                        counts["verified_positive_annotations"] += 1
                    else:
                        counts["matching_spans_not_marked_verified"] += 1
                else:
                    counts["annotations_without_verified_evidence"] += 1

        counts["orphan_annotations"] = sum(
            len(items) for items in annotations_by_document.values()
        )

        totals.update(counts)
        print(f"\n{dataset.upper()}")
        for name in (
            "documents",
            "annotations",
            "verified_positive_annotations",
            "negative_examples",
            "annotations_without_verified_evidence",
            "matching_spans_not_marked_verified",
            "invalid_evidence_spans",
            "negative_examples_with_spans",
            "orphan_annotations",
            "duplicate_document_ids",
            "incorrect_document_hashes",
            "empty_documents",
        ):
            print(f"  {name}: {counts[name]:,}")

    print("\nCOMBINED")
    print(f"  document_records: {totals['documents']:,}")
    print(f"  unique_exact_texts: {len(hashes):,}")
    print(f"  duplicate_text_groups: {sum(n > 1 for n in hashes.values()):,}")
    print(f"  extra_duplicate_records: {sum(n - 1 for n in hashes.values()):,}")

    errors = (
        "invalid_evidence_spans",
        "negative_examples_with_spans",
        "orphan_annotations",
        "duplicate_document_ids",
        "incorrect_document_hashes",
        "empty_documents",
    )
    if any(totals[name] for name in errors):
        raise SystemExit("\nFAIL: Resolve the integrity errors before merging.")

    print("\nPASS: Document and annotation integrity checks passed.")


if __name__ == "__main__":
    main()