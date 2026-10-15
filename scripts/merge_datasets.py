"""Merge datasets, deduplicate exact text, and preserve source links."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "data/processed/datasets/cuad_v2",
    ROOT / "data/processed/datasets/tosdr_v2",
]
OUTPUT = ROOT / "data/processed/datasets/combined_v1"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    # Refuse to overwrite a previous run.
    if OUTPUT.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT}\n"
            "Choose a new OUTPUT folder before running again."
        )

    aliases = {}
    provenance = defaultdict(list)
    counts = Counter()

    # First pass: assign canonical IDs and collect original source records.
    # Document text is streamed rather than retained in memory.
    for folder in INPUTS:
        for document in read_jsonl(folder / "documents.jsonl"):
            original_id = document["document_id"]
            if original_id in aliases:
                raise ValueError(f"Duplicate document ID: {original_id}")

            digest = hashlib.sha256(
                document["original_text"].encode("utf-8")
            ).hexdigest()

            if digest != document["content_sha256"]:
                raise ValueError(f"Incorrect text hash: {original_id}")

            canonical_id = f"merged:{digest}"
            aliases[original_id] = canonical_id

            provenance[canonical_id].append(
                {
                    "original_document_id": original_id,
                    "title": document["title"],
                    "document_type": document["document_type"],
                    "source_type": document["source_type"],
                    "source_name": document.get("source_name"),
                    "metadata": document.get("metadata", {}),
                }
            )
            counts["input_documents"] += 1

    # Validate annotation/evaluation references before creating output.
    for folder in INPUTS:
        for filename in ("annotations.jsonl", "evaluation.jsonl"):
            for item in read_jsonl(folder / filename):
                if item["document_id"] not in aliases:
                    raise ValueError(
                        f"Unknown document in {filename}: "
                        f"{item['document_id']}"
                    )

    OUTPUT.mkdir(parents=True)
    written = set()

    with (OUTPUT / "documents.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for folder in INPUTS:
            for document in read_jsonl(folder / "documents.jsonl"):
                canonical_id = aliases[document["document_id"]]
                if canonical_id in written:
                    continue

                written.add(canonical_id)
                sources = provenance[canonical_id]
                source_datasets = sorted(
                    {
                        source["metadata"]["source_dataset"]
                        for source in sources
                    }
                )

                document["document_id"] = canonical_id
                document["metadata"] = {
                    "source_datasets": source_datasets,
                    "source_records": sources,
                }

                # Text and character positions remain unchanged.
                write_record(stream, document)
                counts["unique_documents"] += 1

    with (OUTPUT / "document_aliases.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for original_id, canonical_id in aliases.items():
            write_record(
                stream,
                {
                    "original_document_id": original_id,
                    "document_id": canonical_id,
                },
            )

    with (
        (OUTPUT / "annotations.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as all_annotations,
        (OUTPUT / "annotations_verified.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as verified_annotations,
    ):
        for folder in INPUTS:
            for item in read_jsonl(folder / "annotations.jsonl"):
                original_id = item["document_id"]
                item["original_document_id"] = original_id
                item["document_id"] = aliases[original_id]
                write_record(all_annotations, item)
                counts["annotations"] += 1

                has_verified_evidence = (
                    item.get("offsets_validated") is True
                    and bool(item.get("evidence_spans"))
                )
                is_negative = (
                    item.get("source_dataset") == "CUAD"
                    and item.get("is_impossible") is True
                    and not item.get("evidence_spans")
                )

                if has_verified_evidence or is_negative:
                    write_record(verified_annotations, item)
                    counts["verified_annotations"] += 1
                else:
                    counts["unverified_annotations"] += 1

    # Preserve existing evaluation cases; do not invent ToS;DR questions.
    with (OUTPUT / "evaluation.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for folder in INPUTS:
            for item in read_jsonl(folder / "evaluation.jsonl"):
                original_id = item["document_id"]
                item["original_document_id"] = original_id
                item["document_id"] = aliases[original_id]
                write_record(stream, item)
                counts["evaluation_cases"] += 1

    counts["duplicate_documents_removed"] = (
        counts["input_documents"] - counts["unique_documents"]
    )

    manifest = {
        "inputs": [str(folder) for folder in INPUTS],
        "deduplication": "SHA-256 of unchanged original_text",
        "counts": dict(counts),
        "source_manifests": [
            json.loads(
                (folder / "source_manifest.json").read_text(encoding="utf-8")
            )
            for folder in INPUTS
        ],
    }

    # Written last: its presence indicates the script reached completion.
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(dict(counts), indent=2))
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()