import csv
import json

from contract_rag.dataset_adapters import (
    adapt_dataset,
    export_dataset,
    iter_contractnli,
    iter_open_terms,
    iter_tosdr,
)


def test_tosdr_adapter_keeps_exact_quote_offsets_and_original_document(tmp_path):
    text = "TERMS\nWe may suspend access after notice.\n"
    quote = "We may suspend access after notice."
    start = text.index(quote)
    documents = tmp_path / "documents.csv"
    with documents.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "name", "url", "text", "service_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "42",
                "name": "Example Terms",
                "url": "https://example.test/terms",
                "text": text,
                "service_id": "example",
            }
        )
    points = tmp_path / "points.csv"
    with points.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "document_id", "quote", "quote_start", "quote_end", "title"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "p1",
                "document_id": "42",
                "quote": quote,
                "quote_start": start,
                "quote_end": start + len(quote) - 1,
                "title": "Suspension",
            }
        )

    row = next(iter_tosdr(documents, points))

    assert row.document.original_text == text
    assert row.document.metadata["source_url"] == "https://example.test/terms"
    span = row.annotations[0]["evidence_spans"][0]
    assert row.document.original_text[span["start_char"] : span["end_char"]] == quote
    assert row.annotations[0]["offsets_validated"] is True


def test_open_terms_adapter_retains_each_version_as_a_separate_record(tmp_path):
    first = tmp_path / "Example Service" / "Terms of Service" / "2024-01-01.md"
    second = tmp_path / "Example Service" / "Terms of Service" / "2025-01-01.md"
    first.parent.mkdir(parents=True)
    first.write_text("Version one terms.", encoding="utf-8")
    second.write_text("Version two terms.", encoding="utf-8")

    rows = list(iter_open_terms(tmp_path))

    assert len(rows) == 2
    assert rows[0].document.document_id != rows[1].document.document_id
    assert rows[0].document.metadata["service_name"] == "Example Service"


def test_contractnli_adapter_maps_sentence_indices_to_character_spans(tmp_path):
    text = "Agreement. The supplier must protect confidential information."
    start = text.index("The supplier")
    end = len(text)
    payload = {
        "documents": [
            {
                "id": 7,
                "file_name": "agreement.txt",
                "text": text,
                "url": "https://example.test/agreement",
                "document_type": "sec-text",
                "spans": [[0, 10], [start, end]],
                "labels": {"H1": {"hypothesis": "Supplier protects confidential information."}},
                "annotation_sets": [
                    {"annotations": {"H1": {"choice": "Entailment", "spans": [1]}}}
                ],
            }
        ]
    }
    path = tmp_path / "contractnli.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    row = next(iter_contractnli(path))
    evidence = row.annotations[0]["evidence_spans"][0]

    assert row.document.original_text[evidence["start_char"] : evidence["end_char"]] == (
        "The supplier must protect confidential information."
    )
    assert row.annotations[0]["label"] == "Entailment"
    assert row.annotations[0]["offsets_validated"] is True


def test_cuad_adapter_exports_documents_separate_from_annotations(tmp_path):
    context = "Section 1. Liability shall not exceed the fees paid."
    answer = "Liability shall not exceed the fees paid."
    start = context.index(answer)
    source = tmp_path / "cuad.json"
    source.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Example Contract",
                        "paragraphs": [
                            {
                                "context": context,
                                "qas": [
                                    {
                                        "id": "example__Cap On Liability",
                                        "question": 'Find "Cap On Liability".',
                                        "is_impossible": False,
                                        "answers": [{"text": answer, "answer_start": start}],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized"

    counts = export_dataset(adapt_dataset("cuad", source), output)

    assert counts == {"document_count": 1, "annotation_count": 1, "evaluation_case_count": 1}
    document = json.loads((output / "documents.jsonl").read_text(encoding="utf-8"))
    annotation = json.loads((output / "annotations.jsonl").read_text(encoding="utf-8"))
    assert document["original_text"] == context
    assert annotation["document_id"] == document["document_id"]
    assert document["original_text"][
        annotation["evidence_spans"][0]["start_char"] : annotation["evidence_spans"][0]["end_char"]
    ] == answer


def test_maud_adapter_accepts_squad_style_json_and_text_directories(tmp_path):
    text = "The buyer may terminate for cause."
    answer = "terminate for cause"
    data_path = tmp_path / "maud.json"
    data_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Merger Agreement",
                        "paragraphs": [
                            {
                                "context": text,
                                "qas": [
                                    {
                                        "id": "q1",
                                        "question": "When may the buyer terminate?",
                                        "answers": [
                                            {"text": answer, "answer_start": text.index(answer)}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    maud_row = next(adapt_dataset("maud", data_path))
    assert maud_row.annotations[0]["source_dataset"] == "MAUD"
    assert maud_row.annotations[0]["offsets_validated"] is True

    text_dir = tmp_path / "mcc"
    text_dir.mkdir()
    (text_dir / "filing.txt").write_text("Company agreement text.", encoding="utf-8")
    mcc_row = next(adapt_dataset("mcc", text_dir))
    assert mcc_row.document.metadata["source_dataset"] == "Stanford Material Contracts Corpus"

    privacy_dir = tmp_path / "opp115"
    privacy_dir.mkdir()
    (privacy_dir / "policy.txt").write_text("Privacy policy text.", encoding="utf-8")
    privacy_row = next(adapt_dataset("opp115", privacy_dir))
    assert privacy_row.document.metadata["source_dataset"] == "OPP-115"
    assert privacy_row.annotations == ()
