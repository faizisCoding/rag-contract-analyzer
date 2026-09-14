import json

import pytest

from contract_rag.evaluation.cuad import (
    create_document_id,
    extract_clause_type,
    load_cuad,
)


def create_test_dataset(tmp_path, *, answer_start: int = 32):
    context = (
        "1. Services\n"
        "The supplier provides services.\n"
        "2. Liability\n"
        "Liability shall not exceed the fees paid."
    )

    answer_text = "Liability shall not exceed the fees paid."
    actual_start = context.index(answer_text)

    payload = {
        "version": "test",
        "data": [
            {
                "title": "Example Agreement",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": "example__Cap On Liability",
                                "question": (
                                    "Highlight the parts related to "
                                    '"Cap On Liability" that should be reviewed.'
                                ),
                                "is_impossible": False,
                                "answers": [
                                    {
                                        "text": answer_text,
                                        "answer_start": (
                                            actual_start if answer_start == 32 else answer_start
                                        ),
                                    }
                                ],
                            },
                            {
                                "id": "example__Termination For Convenience",
                                "question": (
                                    "Highlight the parts related to "
                                    '"Termination For Convenience" '
                                    "that should be reviewed."
                                ),
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    path = tmp_path / "cuad-test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, context


def test_loads_positive_and_negative_annotations(tmp_path):
    path, context = create_test_dataset(tmp_path)

    documents = load_cuad(path)

    assert len(documents) == 1

    document = documents[0]

    assert document.title == "Example Agreement"
    assert document.context == context
    assert document.document_id == create_document_id(
        "Example Agreement",
        context,
    )

    assert len(document.annotations) == 2

    liability = document.annotations[0]
    termination = document.annotations[1]

    assert liability.clause_type == "Cap On Liability"
    assert liability.is_impossible is False
    assert len(liability.answers) == 1

    assert termination.clause_type == "Termination For Convenience"
    assert termination.is_impossible is True
    assert termination.answers == ()


def test_rejects_invalid_answer_offset(tmp_path):
    path, _ = create_test_dataset(tmp_path, answer_start=0)

    with pytest.raises(ValueError, match="Answer-offset mismatch"):
        load_cuad(path)


def test_extracts_clause_from_training_identifier():
    clause_type = extract_clause_type(
        question="Unrecognized question format",
        query_id="agreement__Cap On Liability_3",
    )

    assert clause_type == "Cap On Liability"
