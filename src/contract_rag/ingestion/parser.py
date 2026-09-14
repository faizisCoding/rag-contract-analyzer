import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def calculate_sha256(file_path: Path) -> str:
    """Calculate a stable fingerprint for document deduplication."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Write JSON data using UTF-8 encoding."""

    file_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_contract(
    source_path: str | Path,
    output_directory: str | Path = "data/processed",
) -> dict[str, Any]:
    """Parse a contract and preserve Docling's structured output."""

    source = Path(source_path).resolve()
    output_dir = Path(output_directory).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Contract not found: {source}")

    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {source.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    document_hash = calculate_sha256(source)
    started_at = time.perf_counter()

    converter = DocumentConverter()
    conversion_result = converter.convert(source=str(source))
    document = conversion_result.document

    markdown = document.export_to_markdown()
    structured_document = document.export_to_dict()

    elapsed_seconds = time.perf_counter() - started_at
    page_count = len(document.pages)

    output_stem = f"{source.stem}-{document_hash[:12]}"
    markdown_path = output_dir / f"{output_stem}.md"
    json_path = output_dir / f"{output_stem}.json"
    metadata_path = output_dir / f"{output_stem}.metadata.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    write_json(json_path, structured_document)

    metadata = {
        "source_filename": source.name,
        "source_path": str(source),
        "sha256": document_hash,
        "page_count": page_count,
        "markdown_character_count": len(markdown),
        "processing_seconds": round(elapsed_seconds, 3),
        "docling_version": version("docling"),
        "markdown_output": str(markdown_path),
        "structured_output": str(json_path),
    }

    write_json(metadata_path, metadata)
    return metadata
