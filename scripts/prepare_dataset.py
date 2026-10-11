import argparse
import json
from pathlib import Path

from contract_rag.dataset_adapters import adapt_dataset, export_dataset


DATASETS = (
    "cuad",
    "tosdr",
    "open_terms",
    "contractnli",
    "maud",
    "mcc",
    "opp115",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a legal dataset into canonical documents, annotations, and evaluation JSONL."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset JSON/CSV file or extracted directory.",
    )
    parser.add_argument("--points", help="Optional ToS;DR points.csv file.")
    parser.add_argument(
        "--output",
        help="Output directory (defaults to data/processed/datasets/<dataset>).",
    )
    parser.add_argument(
        "--license-note",
        default=(
            "Review the source dataset and underlying-document terms before redistribution "
            "or commercial use."
        ),
        help="License/provenance note copied into source_manifest.json.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output or f"data/processed/datasets/{args.dataset}").resolve()
    counts = export_dataset(
        adapt_dataset(args.dataset, args.input, points_path=args.points),
        output_dir,
    )
    manifest = {
        "dataset": args.dataset,
        "input": str(Path(args.input).resolve()),
        "points_input": str(Path(args.points).resolve()) if args.points else None,
        "license_note": args.license_note,
        "format_version": 1,
        **counts,
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_dir), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
