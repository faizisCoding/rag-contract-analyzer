import codecs
import hashlib
from pathlib import Path

from contract_rag.models import (
    LegalDocument,
    SourceType,
    create_legal_document,
)


def ingest_txt(
    source_path: str | Path,
    *,
    title: str | None = None,
) -> LegalDocument:
    """Load a UTF-8 TXT file without changing its newlines or spacing."""

    source = Path(source_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Text document not found: {source}")

    if not source.is_file():
        raise ValueError(f"Text document is not a file: {source}")

    if source.suffix.lower() != ".txt":
        extension = source.suffix or "<no extension>"
        raise ValueError(f"Expected a .txt file, received: {extension}")

    raw_bytes = source.read_bytes()

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UnicodeError(f"TXT input must be UTF-8 encoded: {source.name}") from error

    encoding = "utf-8-sig" if raw_bytes.startswith(codecs.BOM_UTF8) else "utf-8"

    return create_legal_document(
        text=text,
        title=title or source.stem,
        source_type=SourceType.TXT,
        source_name=source.name,
        metadata={
            "encoding": encoding,
            "size_bytes": len(raw_bytes),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
    )


def ingest_pasted_text(
    text: str,
    *,
    title: str = "Pasted legal document",
) -> LegalDocument:
    """Create a canonical document from text submitted by the future UI."""

    if not isinstance(text, str):
        raise TypeError("Pasted text must be a string")

    return create_legal_document(
        text=text,
        title=title,
        source_type=SourceType.PASTED_TEXT,
        source_name=None,
        metadata={"input_method": "paste"},
    )
