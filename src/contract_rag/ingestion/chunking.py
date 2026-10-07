import hashlib
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from contract_rag.models import (
    LegalDocument,
    SourceLocation,
    normalize_for_search,
)

CHUNKER_VERSION = "section-v1"

_NUMBERED_HEADING = re.compile(
    r"^(?:(?:section|article)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+)"
    r"(?:[.):]|\s|$)"
    r"|\d+(?:\.\d+)*[.)]?\s+\S)",
    re.IGNORECASE,
)
_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")
_LINE_BREAK = re.compile(r"\r?\n")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LegalChunk:
    chunk_id: str
    chunk_index: int
    text: str
    search_text: str
    location: SourceLocation


def _is_heading(line: str) -> bool:
    candidate = line.strip()

    if not candidate or len(candidate) > 120:
        return False

    # Avoid treating ordinary numbered sentences as headings.
    if candidate.endswith((".", ";", "?", "!")):
        return False

    if _NUMBERED_HEADING.match(candidate):
        return True

    return candidate.isupper() and any(character.isalpha() for character in candidate)


def _section_spans(
    text: str,
) -> list[tuple[int, int, str | None]]:
    headings: list[tuple[int, str]] = []
    offset = 0

    for line in text.splitlines(keepends=True):
        if _is_heading(line):
            headings.append((offset, line.strip()))
        offset += len(line)

    if not headings:
        return [(0, len(text), None)]

    sections: list[tuple[int, int, str | None]] = []

    if headings[0][0] > 0:
        sections.append((0, headings[0][0], None))

    for index, (start, title) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        sections.append((start, end, title))

    return sections


def _choose_end(
    text: str,
    start: int,
    section_end: int,
    max_chars: int,
) -> int:
    limit = min(start + max_chars, section_end)

    if limit == section_end:
        return limit

    # Prefer boundaries in the latter half of the available window.
    minimum = start + max_chars // 2
    window = text[start:limit]

    for pattern in (_PARAGRAPH_BREAK, _LINE_BREAK, _WHITESPACE):
        candidates = [
            start + match.end()
            for match in pattern.finditer(window)
            if start + match.end() >= minimum
            and not (text[start + match.end() - 1] == "\r" and text[start + match.end()] == "\n")
        ]

        if candidates:
            return candidates[-1]

    # Keep Windows CRLF pairs together during a hard split.
    if text[limit - 1] == "\r" and text[limit] == "\n":
        return limit - 1

    return limit


def chunk_document(
    document: LegalDocument,
    *,
    max_chars: int = 1800,
) -> list[LegalChunk]:
    """Partition original_text into exact, contiguous source spans."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 2:
        raise ValueError("max_chars must be at least 2")

    chunks: list[LegalChunk] = []

    for section_start, section_end, section_title in _section_spans(document.original_text):
        start = section_start

        while start < section_end:
            end = _choose_end(
                document.original_text,
                start,
                section_end,
                max_chars,
            )
            location = replace(
                document.locate(start, end),
                section_title=section_title,
            )
            text = document.quote(location)

            identity = (
                f"{CHUNKER_VERSION}\0{document.document_id}\0"
                f"{document.content_sha256}\0{max_chars}\0{start}\0{end}"
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

            chunks.append(
                LegalChunk(
                    chunk_id=f"chunk-{chunk_id}",
                    chunk_index=len(chunks),
                    text=text,
                    search_text=normalize_for_search(text),
                    location=location,
                )
            )
            start = end

    return chunks


def chunk_to_dict(chunk: LegalChunk) -> dict[str, Any]:
    """Convert a chunk into a JSON-serializable record."""

    return {
        "chunker_version": CHUNKER_VERSION,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.location.document_id,
        "text": chunk.text,
        "search_text": chunk.search_text,
        "location": asdict(chunk.location),
    }
