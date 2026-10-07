import codecs

import pytest

from contract_rag.ingestion.text import (
    ingest_pasted_text,
    ingest_txt,
)
from contract_rag.models import SourceType


def test_pasted_text_preserves_source_and_builds_exact_citation():
    text = "1. Terms\r\nLiability\u00a0is unlimited.\r\nEnd."

    document = ingest_pasted_text(
        text,
        title="Example Terms",
    )

    assert document.source_type is SourceType.PASTED_TEXT
    assert document.original_text == text
    assert document.search_text == "1. Terms Liability is unlimited. End."
    assert document.line_count == 3

    quote = "Liability\u00a0is unlimited."
    start_char = text.index(quote)

    location = document.locate(
        start_char,
        start_char + len(quote),
    )

    assert location.start_line == 2
    assert location.end_line == 2
    assert document.quote(location) == quote


def test_txt_ingestion_accepts_utf8_bom_and_preserves_crlf(
    tmp_path,
):
    text = "LICENSE AGREEMENT\r\n1. Grant of License\r\nLimited use only."

    source = tmp_path / "license.txt"

    source.write_bytes(codecs.BOM_UTF8 + text.encode("utf-8"))

    document = ingest_txt(source)

    assert document.title == "license"
    assert document.source_type is SourceType.TXT
    assert document.original_text == text
    assert document.metadata["encoding"] == "utf-8-sig"
    assert document.line_count == 3


def test_same_exact_text_produces_same_document_id(
    tmp_path,
):
    text = "Terms and conditions"
    source = tmp_path / "terms.txt"

    source.write_bytes(text.encode("utf-8"))

    uploaded = ingest_txt(source)
    pasted = ingest_pasted_text(text)

    assert uploaded.document_id == pasted.document_id
    assert uploaded.content_sha256 == pasted.content_sha256


def test_txt_ingestion_rejects_non_utf8_input(
    tmp_path,
):
    source = tmp_path / "legacy.txt"
    source.write_bytes(b"License fee: \x96 100")

    with pytest.raises(
        UnicodeError,
        match="must be UTF-8",
    ):
        ingest_txt(source)


def test_pasted_text_rejects_blank_input():
    with pytest.raises(
        ValueError,
        match="original_text cannot be empty",
    ):
        ingest_pasted_text("   \n\t")
