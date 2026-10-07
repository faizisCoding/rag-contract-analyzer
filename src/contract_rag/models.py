import hashlib
import unicodedata
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    PASTED_TEXT = "pasted_text"


class DocumentType(StrEnum):
    COMMERCIAL_CONTRACT = "commercial_contract"
    TERMS_OF_SERVICE = "terms_of_service"
    LICENSE_AGREEMENT = "license_agreement"
    EULA = "eula"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PageSpan:
    """An exact, zero-based character span belonging to one PDF/DOCX page."""

    page_number: int
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("Page spans must be non-empty and use valid offsets")


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A citation into a LegalDocument; end_char is exclusive."""

    document_id: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    start_page: int | None = None
    end_page: int | None = None
    section_title: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("Source locations must be non-empty and use valid offsets")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("Source locations must use valid one-based line numbers")
        if self.start_page is not None and self.start_page < 1:
            raise ValueError("start_page must be one-based")
        if self.end_page is not None and (
            self.start_page is None or self.end_page < self.start_page
        ):
            raise ValueError("end_page must not precede start_page")


@dataclass(frozen=True, slots=True)
class LegalDocument:
    document_id: str
    title: str
    source_type: SourceType
    original_text: str
    search_text: str
    content_sha256: str
    line_start_offsets: tuple[int, ...]
    source_name: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    page_spans: tuple[PageSpan, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not self.title.strip():
            raise ValueError("title cannot be empty")
        if not self.original_text.strip():
            raise ValueError("original_text cannot be empty")

        expected_hash = hashlib.sha256(self.original_text.encode("utf-8")).hexdigest()

        if self.content_sha256 != expected_hash:
            raise ValueError("content_sha256 does not match original_text")

        if self.search_text != normalize_for_search(self.original_text):
            raise ValueError("search_text is not the normalized copy of original_text")

        if self.line_start_offsets != build_line_start_offsets(self.original_text):
            raise ValueError("line_start_offsets do not match original_text")

        previous_page = 0
        previous_end = 0

        for page in self.page_spans:
            if page.page_number <= previous_page:
                raise ValueError("page_spans must have increasing page numbers")
            if page.start_char < previous_end:
                raise ValueError("page_spans cannot overlap")
            if page.end_char > len(self.original_text):
                raise ValueError("page span exceeds original_text")

            previous_page = page.page_number
            previous_end = page.end_char

    @property
    def character_count(self) -> int:
        return len(self.original_text)

    @property
    def line_count(self) -> int:
        return len(self.line_start_offsets)

    def locate(
        self,
        start_char: int,
        end_char: int,
    ) -> SourceLocation:
        """Build a validated citation for an exact, non-empty text span."""

        if start_char < 0 or end_char <= start_char:
            raise ValueError("Citation spans must be non-empty and use valid offsets")

        if end_char > len(self.original_text):
            raise ValueError("Citation span exceeds original_text")

        return SourceLocation(
            document_id=self.document_id,
            start_char=start_char,
            end_char=end_char,
            start_line=self._line_for_offset(start_char),
            end_line=self._line_for_offset(end_char - 1),
            start_page=self._page_for_offset(start_char),
            end_page=self._page_for_offset(end_char - 1),
        )

    def quote(self, location: SourceLocation) -> str:
        """Return the exact source text covered by a location."""

        if location.document_id != self.document_id:
            raise ValueError("Citation belongs to a different document")

        if (
            location.start_char < 0
            or location.end_char <= location.start_char
            or location.end_char > len(self.original_text)
        ):
            raise ValueError("Citation span exceeds original_text")

        return self.original_text[location.start_char : location.end_char]

    def _line_for_offset(self, char_offset: int) -> int:
        return bisect_right(self.line_start_offsets, char_offset)

    def _page_for_offset(
        self,
        char_offset: int,
    ) -> int | None:
        for page in self.page_spans:
            if page.start_char <= char_offset < page.end_char:
                return page.page_number

        return None


def build_line_start_offsets(text: str) -> tuple[int, ...]:
    """Return the zero-based start offset of every line."""

    return (
        0,
        *(index + 1 for index, character in enumerate(text) if character == "\n"),
    )


def normalize_for_search(text: str) -> str:
    """Normalize a copy for retrieval without changing original_text."""

    unicode_normalized = unicodedata.normalize("NFKC", text)
    return " ".join(unicode_normalized.split())


def create_legal_document(
    *,
    text: str,
    title: str,
    source_type: SourceType,
    source_name: str | None = None,
    document_type: DocumentType = DocumentType.UNKNOWN,
    document_id: str | None = None,
    page_spans: tuple[PageSpan, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> LegalDocument:
    """Create the canonical representation used by later RAG stages."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    resolved_document_id = document_id or f"legal-{content_sha256[:16]}"

    return LegalDocument(
        document_id=resolved_document_id,
        title=title,
        source_type=source_type,
        original_text=text,
        search_text=normalize_for_search(text),
        content_sha256=content_sha256,
        line_start_offsets=build_line_start_offsets(text),
        source_name=source_name,
        document_type=document_type,
        page_spans=page_spans,
        metadata=dict(metadata or {}),
    )


def legal_document_to_dict(
    document: LegalDocument,
) -> dict[str, Any]:
    """Convert a LegalDocument into a JSON-serializable dictionary."""

    return {
        "document_id": document.document_id,
        "title": document.title,
        "document_type": document.document_type.value,
        "source_type": document.source_type.value,
        "source_name": document.source_name,
        "original_text": document.original_text,
        "search_text": document.search_text,
        "content_sha256": document.content_sha256,
        "character_count": document.character_count,
        "line_count": document.line_count,
        "line_start_offsets": list(document.line_start_offsets),
        "page_spans": [
            {
                "page_number": page.page_number,
                "start_char": page.start_char,
                "end_char": page.end_char,
            }
            for page in document.page_spans
        ],
        "metadata": dict(document.metadata),
    }
