"""Source adapters that normalize legal datasets into the project document model.

The adapters preserve original text and source provenance. Annotation exports are
separate from document text so labels and questions cannot pollute RAG retrieval.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contract_rag.evaluation.cuad import load_cuad
from contract_rag.models import (
    DocumentType,
    LegalDocument,
    SourceType,
    create_legal_document,
    legal_document_to_dict,
)


@dataclass(frozen=True, slots=True)
class DatasetRow:
    document: LegalDocument
    annotations: tuple[dict[str, Any], ...] = ()
    evaluation_cases: tuple[dict[str, Any], ...] = ()


def _stable_id(source: str, source_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", source_id).strip("-")[:80]
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{slug or 'record'}:{digest}"


def _document(
    *,
    source: str,
    source_id: str,
    title: str,
    text: str,
    document_type: DocumentType,
    source_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> LegalDocument:
    merged = {"source_dataset": source, "source_record_id": source_id}
    merged.update(metadata or {})
    return create_legal_document(
        text=text,
        title=title or source_id,
        source_type=SourceType.TXT,
        source_name=source_name,
        document_type=document_type,
        document_id=_stable_id(source, source_id),
        metadata=merged,
    )


def export_dataset(rows: Iterable[DatasetRow], output_dir: str | Path) -> dict[str, int]:
    """Write canonical documents, annotations, and evaluation cases as JSONL."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    documents_path = target / "documents.jsonl"
    annotations_path = target / "annotations.jsonl"
    evaluations_path = target / "evaluation.jsonl"

    document_count = annotation_count = evaluation_count = 0
    with (
        documents_path.open("w", encoding="utf-8", newline="\n") as documents_file,
        annotations_path.open("w", encoding="utf-8", newline="\n") as annotations_file,
        evaluations_path.open("w", encoding="utf-8", newline="\n") as evaluations_file,
    ):
        for row in rows:
            documents_file.write(
                json.dumps(legal_document_to_dict(row.document), ensure_ascii=False) + "\n"
            )
            document_count += 1
            for annotation in row.annotations:
                annotations_file.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                annotation_count += 1
            for case in row.evaluation_cases:
                evaluations_file.write(json.dumps(case, ensure_ascii=False) + "\n")
                evaluation_count += 1

    return {
        "document_count": document_count,
        "annotation_count": annotation_count,
        "evaluation_case_count": evaluation_count,
    }


def iter_cuad(path: str | Path) -> Iterator[DatasetRow]:
    """Adapt CUAD's SQuAD-style JSON, retaining validated answer offsets."""

    for source_doc in load_cuad(path, validate_spans=True):
        document = _document(
            source="CUAD",
            source_id=source_doc.document_id,
            title=source_doc.title,
            text=source_doc.context,
            document_type=DocumentType.COMMERCIAL_CONTRACT,
            source_name=source_doc.title,
            metadata={"source_format": "squad_json"},
        )
        annotations: list[dict[str, Any]] = []
        evaluation_cases: list[dict[str, Any]] = []
        for item in source_doc.annotations:
            spans = [
                {
                    "start_char": answer.answer_start,
                    "end_char": answer.answer_end,
                    "text": answer.text,
                }
                for answer in item.answers
            ]
            annotation = {
                "annotation_id": item.query_id,
                "document_id": document.document_id,
                "source_dataset": "CUAD",
                "task": "clause_extraction",
                "label": item.clause_type,
                "question": item.question,
                "is_impossible": item.is_impossible,
                "evidence_spans": spans,
                "offsets_validated": True,
            }
            annotations.append(annotation)
            evaluation_cases.append(annotation.copy())
        yield DatasetRow(document, tuple(annotations), tuple(evaluation_cases))


def _read_csv(path: str | Path) -> Iterator[dict[str, str]]:
    # Full policy documents can exceed Python's default CSV field limit.
    csv.field_size_limit(100_000_000)

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def iter_tosdr(
    documents_csv: str | Path,
    points_csv: str | Path | None = None,
) -> Iterator[DatasetRow]:
    """Adapt ToS;DR documents and optional quoted-point annotations.

    Source offsets are checked against the quote using either end convention.
    Output spans always use an exclusive end. Unique exact quote matches can
    recover positions when source offsets do not match.
    """

    points_by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    if points_csv:
        for point in _read_csv(points_csv):
            document_id = (point.get("document_id") or "").strip()
            points_by_document[document_id].append(point)

    for record in _read_csv(documents_csv):
        source_id = (record.get("id") or "").strip()
        text = record.get("text") or ""
        if not source_id or not text.strip():
            continue

        agreement_name = (record.get("name") or "").lower()
        inferred_type = (
            DocumentType.OTHER
            if "privacy" in agreement_name
            else DocumentType.EULA
            if "license" in agreement_name or "eula" in agreement_name
            else DocumentType.TERMS_OF_SERVICE
        )

        document = _document(
            source="ToS;DR",
            source_id=source_id,
            title=record.get("name") or f"ToS;DR document {source_id}",
            text=text,
            document_type=inferred_type,
            source_name="documents.csv",
            metadata={
                "source_url": record.get("url") or None,
                "service_id": record.get("service_id") or None,
                "created_at": record.get("created_at") or None,
                "updated_at": record.get("updated_at") or None,
                "source_format": "tosdr_documents_csv",
            },
        )

        annotations: list[dict[str, Any]] = []
        for point in points_by_document.get(source_id, []):
            quote = point.get("quote_text") or point.get("quote") or ""
            start_raw = point.get("quote_start", "")
            end_raw = point.get("quote_end", "")

            start, end_exclusive = -1, -1
            exact = False
            match_method = "unmatched"

            try:
                source_start = int(start_raw)
                source_end = int(end_raw)
            except (TypeError, ValueError):
                source_start, source_end = -1, -1

            if quote.strip():
                # Prefer exclusive source offsets when the quote matches.
                if (
                    0 <= source_start < source_end <= len(text)
                    and text[source_start:source_end] == quote
                ):
                    start, end_exclusive = source_start, source_end
                    exact = True
                    match_method = "source_exclusive"

                # Support inclusive offsets only when verified by exact text.
                elif (
                    0 <= source_start <= source_end < len(text)
                    and text[source_start : source_end + 1] == quote
                ):
                    start, end_exclusive = source_start, source_end + 1
                    exact = True
                    match_method = "source_inclusive"

                else:
                    position = text.find(quote)
                    if position >= 0:
                        if text.find(quote, position + 1) == -1:
                            start = position
                            end_exclusive = position + len(quote)
                            exact = True
                            match_method = "unique_exact_search"
                        else:
                            match_method = "ambiguous"
            else:
                match_method = "missing_quote"

            annotations.append(
                {
                    "annotation_id": str(point.get("id") or point.get("old_id") or ""),
                    "document_id": document.document_id,
                    "source_dataset": "ToS;DR",
                    "task": "terms_point",
                    "label": (point.get("title") or point.get("topic_id") or "unspecified"),
                    "quote": quote,
                    "evidence_spans": (
                        [
                            {
                                "start_char": start,
                                "end_char": end_exclusive,
                                "text": quote,
                            }
                        ]
                        if exact
                        else []
                    ),
                    "offsets_validated": exact,
                    "source_quote_start": start_raw,
                    "source_quote_end": end_raw,
                    "evidence_match_method": match_method,
                }
            )

        yield DatasetRow(document, tuple(annotations))


def iter_open_terms(root: str | Path) -> Iterator[DatasetRow]:
    """Adapt an Open Terms Archive versions checkout (one Markdown per version)."""

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"Open Terms Archive directory not found: {base}")
    for path in sorted(base.rglob("*.md")):
        relative = path.relative_to(base)
        if path.name.lower() in {"readme.md", "license.md", "contributing.md"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            continue
        parts = relative.parts
        service = parts[0] if len(parts) > 1 else path.parent.name
        agreement_type = parts[1] if len(parts) > 2 else "Terms"
        source_id = relative.as_posix()
        title = f"{service} — {agreement_type} — {path.stem}"
        kind = (
            DocumentType.TERMS_OF_SERVICE
            if "privacy" not in agreement_type.lower()
            else DocumentType.OTHER
        )
        yield DatasetRow(
            _document(
                source="Open Terms Archive",
                source_id=source_id,
                title=title,
                text=text,
                document_type=kind,
                source_name=relative.as_posix(),
                metadata={
                    "service_name": service,
                    "agreement_type": agreement_type,
                    "version": path.stem,
                    "source_path": relative.as_posix(),
                    "source_format": "ota_markdown_version",
                },
            )
        )


def iter_contractnli(path: str | Path) -> Iterator[DatasetRow]:
    """Adapt ContractNLI JSON documents, hypotheses, labels, and evidence spans."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("ContractNLI JSON must contain a top-level 'documents' list")
    for record in raw_documents:
        source_id = str(record.get("id", ""))
        text = str(record.get("text", ""))
        if not source_id or not text.strip():
            continue
        document = _document(
            source="ContractNLI",
            source_id=source_id,
            title=record.get("file_name") or f"ContractNLI {source_id}",
            text=text,
            document_type=DocumentType.COMMERCIAL_CONTRACT,
            source_name=record.get("file_name") or "ContractNLI JSON",
            metadata={
                "source_url": record.get("url"),
                "source_document_type": record.get("document_type"),
                "source_format": "contractnli_json",
            },
        )
        labels = record.get("labels", {})
        annotations: list[dict[str, Any]] = []
        for set_index, annotation_set in enumerate(record.get("annotation_sets", [])):
            for hypothesis_id, value in annotation_set.get("annotations", {}).items():
                spans: list[dict[str, Any]] = []
                for span_index in value.get("spans", []):
                    start, end = record["spans"][int(span_index)]
                    if start < 0 or end <= start or end > len(text):
                        raise ValueError(
                            f"ContractNLI evidence span is outside document {source_id}"
                        )
                    evidence_text = text[start:end]
                    spans.append({"start_char": start, "end_char": end, "text": evidence_text})
                hypothesis = labels.get(hypothesis_id, {}).get("hypothesis", hypothesis_id)
                annotations.append(
                    {
                        "annotation_id": f"{source_id}:{set_index}:{hypothesis_id}",
                        "document_id": document.document_id,
                        "source_dataset": "ContractNLI",
                        "task": "document_nli",
                        "label": value.get("choice"),
                        "question": hypothesis,
                        "evidence_spans": spans,
                        "offsets_validated": all(
                            text[span["start_char"] : span["end_char"]] == span["text"]
                            for span in spans
                        ),
                    }
                )
        yield DatasetRow(document, tuple(annotations), tuple(annotations))


def iter_squad_json(path: str | Path, *, source: str) -> Iterator[DatasetRow]:
    """Adapt SQuAD-style legal QA JSON (used by CUAD and compatible exports)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for record in payload.get("data", []):
        title = str(record.get("title", "Untitled agreement"))
        for paragraph_index, paragraph in enumerate(record.get("paragraphs", [])):
            text = str(paragraph.get("context", ""))
            if not text.strip():
                continue
            source_id = f"{title}:{paragraph_index}"
            document = _document(
                source=source,
                source_id=source_id,
                title=title,
                text=text,
                document_type=DocumentType.COMMERCIAL_CONTRACT,
                source_name=title,
                metadata={"source_format": "squad_json"},
            )
            annotations = []
            for item in paragraph.get("qas", []):
                answers = []
                for answer in item.get("answers", []):
                    start = int(answer["answer_start"])
                    answer_text = str(answer["text"])
                    end = start + len(answer_text)
                    if start < 0 or end > len(text) or text[start:end] != answer_text:
                        raise ValueError(f"Invalid answer offsets for {item.get('id')}")
                    answers.append({"start_char": start, "end_char": end, "text": answer_text})
                annotations.append(
                    {
                        "annotation_id": str(item.get("id", "")),
                        "document_id": document.document_id,
                        "source_dataset": source,
                        "task": "contract_qa",
                        "label": item.get("clause_type"),
                        "question": item.get("question", ""),
                        "is_impossible": bool(item.get("is_impossible", False)),
                        "evidence_spans": answers,
                        "offsets_validated": True,
                    }
                )
            yield DatasetRow(document, tuple(annotations), tuple(annotations))


def iter_text_directory(
    root: str | Path,
    *,
    source: str,
    document_type: DocumentType = DocumentType.UNKNOWN,
) -> Iterator[DatasetRow]:
    """Adapt plain TXT/Markdown documents for MCC or OPP-115 text exports."""

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {base}")
    for path in sorted((*base.rglob("*.txt"), *base.rglob("*.md"))):
        relative = path.relative_to(base)
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            continue
        yield DatasetRow(
            _document(
                source=source,
                source_id=relative.as_posix(),
                title=path.stem,
                text=text,
                document_type=document_type,
                source_name=relative.as_posix(),
                metadata={"source_path": relative.as_posix(), "source_format": "text_file"},
            )
        )


def adapt_dataset(
    source: str,
    input_path: str | Path,
    *,
    points_path: str | Path | None = None,
) -> Iterator[DatasetRow]:
    """Route one named source to its adapter."""

    key = source.lower().replace("-", "_")
    if key == "cuad":
        return iter_cuad(input_path)
    if key in {"tosdr", "tos_dr"}:
        return iter_tosdr(input_path, points_path)
    if key in {"open_terms", "opentermsarchive", "ota"}:
        return iter_open_terms(input_path)
    if key in {"contractnli", "contract_nli"}:
        return iter_contractnli(input_path)
    if key == "maud":
        return iter_squad_json(input_path, source="MAUD")
    if key in {"mcc", "material_contracts_corpus"}:
        return iter_text_directory(
            input_path,
            source="Stanford Material Contracts Corpus",
            document_type=DocumentType.COMMERCIAL_CONTRACT,
        )
    if key in {"opp115", "opp_115"}:
        return iter_text_directory(
            input_path,
            source="OPP-115",
            document_type=DocumentType.OTHER,
        )
    raise ValueError(f"Unsupported dataset adapter: {source}")
