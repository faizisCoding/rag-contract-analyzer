import argparse
import json

from contract_rag.ingestion.parser import parse_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a legal contract using Docling.")
    parser.add_argument(
        "contract",
        help="Path to a PDF or DOCX contract.",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory for parsed output.",
    )

    arguments = parser.parse_args()
    metadata = parse_contract(arguments.contract, arguments.output)

    print("\nContract parsed successfully:\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
