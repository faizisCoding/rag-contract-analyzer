import pytest

from contract_rag.ingestion.chunking import chunk_document
from contract_rag.models import (
    PageSpan,
    SourceType,
    create_legal_document,
)


def make_document(text: str):
    return create_legal_document(
        text=text,
        title="Test agreement",
        source_type=SourceType.PASTED_TEXT,
    )


def test_chunks_preserve_exact_text_offsets_and_crlf():
    text = (
        "1. Liability\r\n"
        + "Liability\u00a0is limited to fees paid. " * 12
        + "\r\n\r\n"
        + "2. Termination\r\nEither party may terminate.\r\n"
    )
    document = make_document(text)
    chunks = chunk_document(document, max_chars=100)

    assert "".join(chunk.text for chunk in chunks) == text

    previous_end = 0
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.location.start_char == previous_end
        assert chunk.text == document.quote(chunk.location)
        assert 0 < len(chunk.text) <= 100
        assert not chunk.text.endswith("\r")
        previous_end = chunk.location.end_char

    assert previous_end == len(text)


def test_detected_sections_remain_separate():
    document = make_document(
        "1. Liability\nFees cap applies.\n\n2. Termination\nEither party may terminate."
    )

    chunks = chunk_document(document, max_chars=1000)

    assert len(chunks) == 2
    assert chunks[0].location.section_title == "1. Liability"
    assert chunks[1].location.section_title == "2. Termination"
    assert "2. Termination" not in chunks[0].text
    assert chunks[1].location.start_line == 4


def test_long_unbroken_text_is_split_without_loss():
    text = "x" * 251
    chunks = chunk_document(make_document(text), max_chars=100)

    assert [len(chunk.text) for chunk in chunks] == [100, 100, 51]
    assert "".join(chunk.text for chunk in chunks) == text


def test_chunk_ids_are_deterministic_and_distinguish_repeated_text():
    document = make_document("x" * 200)

    first = chunk_document(document, max_chars=100)
    second = chunk_document(document, max_chars=100)

    assert first == second
    assert first[0].text == first[1].text
    assert first[0].chunk_id != first[1].chunk_id


def test_prefers_paragraph_boundary():
    first_paragraph = "a" * 60 + "\n\n"
    document = make_document(first_paragraph + "b" * 60)

    chunks = chunk_document(document, max_chars=100)

    assert chunks[0].text == first_paragraph
    assert chunks[1].text == "b" * 60


def test_page_locations_are_preserved():
    document = create_legal_document(
        text="abcdefghij",
        title="Paged agreement",
        source_type=SourceType.PDF,
        page_spans=(
            PageSpan(page_number=1, start_char=0, end_char=5),
            PageSpan(page_number=2, start_char=5, end_char=10),
        ),
    )

    chunks = chunk_document(document, max_chars=5)

    assert chunks[0].location.start_page == 1
    assert chunks[0].location.end_page == 1
    assert chunks[1].location.start_page == 2
    assert chunks[1].location.end_page == 2


def test_whitespace_preamble_is_preserved():
    text = "\r\n \r\n1. Terms\r\nLimited use only."
    chunks = chunk_document(make_document(text), max_chars=100)

    assert "".join(chunk.text for chunk in chunks) == text
    assert chunks[0].search_text == ""
    assert chunks[1].location.section_title == "1. Terms"


@pytest.mark.parametrize("max_chars", [0, 1, -10])
def test_rejects_invalid_chunk_size(max_chars):
    with pytest.raises(ValueError, match="at least 2"):
        chunk_document(make_document("Example terms"), max_chars=max_chars)
