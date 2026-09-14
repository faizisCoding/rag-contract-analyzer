import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TARGET_CLAUSE_TYPES = frozenset(
    {
        "Termination For Convenience",
        "Post-Termination Services",
        "Notice Period To Terminate Renewal",
        "Uncapped Liability",
        "Cap On Liability",
        "Liquidated Damages",
    }
)


@dataclass(frozen=True, slots=True)
class CuadAnswer:
    text: str
    answer_start: int

    @property
    def answer_end(self) -> int:
        return self.answer_start + len(self.text)


@dataclass(frozen=True, slots=True)
class CuadAnnotation:
    query_id: str
    clause_type: str
    question: str
    is_impossible: bool
    answers: tuple[CuadAnswer, ...]


@dataclass(frozen=True, slots=True)
class CuadDocument:
    document_id: str
    title: str
    context: str
    annotations: tuple[CuadAnnotation, ...]


def create_document_id(title: str, context: str) -> str:
    """Create a stable document ID from the title and contract text."""

    content = f"{title}\0{context}".encode()
    digest = hashlib.sha256(content).hexdigest()
    return f"cuad-{digest[:16]}"


def extract_clause_type(question: str, query_id: str) -> str:
    """Extract CUAD's clause category from a question or query ID."""

    match = re.search(
        r'related to\s+["“]([^"”]+)["”]',
        question,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    identifier_suffix = query_id.rsplit("__", maxsplit=1)[-1]
    return re.sub(r"_\d+$", "", identifier_suffix).strip()


def validate_answer_span(
    context: str,
    answer: CuadAnswer,
    *,
    document_title: str,
    query_id: str,
) -> None:
    """Verify that CUAD's annotated text matches its character offsets."""

    if answer.answer_start < 0:
        raise ValueError(f"Negative answer offset in {document_title!r}, query {query_id!r}")

    if answer.answer_end > len(context):
        raise ValueError(f"Answer exceeds context length in {document_title!r}, query {query_id!r}")

    observed_text = context[answer.answer_start : answer.answer_end]

    if observed_text != answer.text:
        raise ValueError(
            "Answer-offset mismatch:\n"
            f"Document: {document_title}\n"
            f"Query: {query_id}\n"
            f"Expected: {answer.text!r}\n"
            f"Observed: {observed_text!r}\n"
            f"Start: {answer.answer_start}"
        )


def load_cuad(
    dataset_path: str | Path,
    *,
    validate_spans: bool = True,
) -> list[CuadDocument]:
    """Load CUAD JSON into structured Python objects."""

    path = Path(dataset_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"CUAD dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        payload: dict[str, Any] = json.load(file)

    raw_documents = payload.get("data")

    if not isinstance(raw_documents, list):
        raise TypeError("Invalid CUAD file: top-level 'data' must be a list")

    documents: list[CuadDocument] = []

    for raw_document in raw_documents:
        title = str(raw_document["title"]).strip()
        paragraphs = raw_document.get("paragraphs", [])

        if not paragraphs:
            raise ValueError(f"Document has no paragraphs: {title}")

        for paragraph_index, paragraph in enumerate(paragraphs):
            context = str(paragraph["context"])
            document_id = create_document_id(title, context)
            annotations: list[CuadAnnotation] = []

            for raw_qa in paragraph.get("qas", []):
                query_id = str(raw_qa["id"])
                question = str(raw_qa["question"]).strip()
                is_impossible = bool(raw_qa.get("is_impossible", False))

                answers = tuple(
                    CuadAnswer(
                        text=str(raw_answer["text"]),
                        answer_start=int(raw_answer["answer_start"]),
                    )
                    for raw_answer in raw_qa.get("answers", [])
                )

                if is_impossible and answers:
                    raise ValueError(f"Impossible query contains answers: {query_id}")

                if not is_impossible and not answers:
                    raise ValueError(f"Positive query has no answers: {query_id}")

                if validate_spans:
                    for answer in answers:
                        validate_answer_span(
                            context,
                            answer,
                            document_title=title,
                            query_id=query_id,
                        )

                annotations.append(
                    CuadAnnotation(
                        query_id=query_id,
                        clause_type=extract_clause_type(question, query_id),
                        question=question,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                )

            if len(paragraphs) > 1:
                document_id = f"{document_id}-p{paragraph_index}"

            documents.append(
                CuadDocument(
                    document_id=document_id,
                    title=title,
                    context=context,
                    annotations=tuple(annotations),
                )
            )

    return documents


def iter_target_annotations(
    documents: Iterable[CuadDocument],
    *,
    clause_types: frozenset[str] = TARGET_CLAUSE_TYPES,
) -> Iterable[tuple[CuadDocument, CuadAnnotation]]:
    """Yield only annotations relevant to the initial project scope."""

    for document in documents:
        for annotation in document.annotations:
            if annotation.clause_type in clause_types:
                yield document, annotation


def answer_to_dict(answer: CuadAnswer) -> dict[str, Any]:
    result = asdict(answer)
    result["answer_end"] = answer.answer_end
    return result
