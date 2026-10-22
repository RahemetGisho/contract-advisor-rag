import json
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from src.ingestion.load_cuad import load_raw_records
from src.ingestion.schema import ContractQARecord
from src.ingestion.validate import validate_records
from src.processing.metadata import Chunk, build_all_chunks, build_contract_documents
from src.utils.config import load_config, resolve_path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Chunks shorter than this are almost always page numbers, footers, or a stray
# sentence tail rather than clause content. Counted, not removed: whether to
# drop or merge them changes the index and every retrieval metric.
TINY_CHUNK_CHARS = 50


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write via a temp file so an interrupted run never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _summary(values: list[int | float]) -> dict:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 1),
        "max": max(values),
    }


def count_span_mismatches(records: list[ContractQARecord]) -> int:

    bad = 0
    for r in records:
        for text, start in zip(r.answer_texts, r.answer_starts, strict=True):
            if r.context[start : start + len(text)] != text:
                bad += 1
    return bad


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def answer_coverage(records: list[ContractQARecord], chunks: list[Chunk]) -> dict:

    lower: dict[str, list[str]] = {}
    normalized: dict[str, list[str]] = {}
    for c in chunks:
        lower.setdefault(c.contract_id, []).append(c.text.lower())
        normalized.setdefault(c.contract_id, []).append(_normalize(c.text))

    answerable = [r for r in records if not r.is_impossible and r.answer_texts]
    literal = sum(
        any(
            a.strip().lower() in t
            for a in r.answer_texts
            for t in lower.get(r.contract_id, [])
        )
        for r in answerable
    )
    norm = sum(
        any(
            _normalize(a) in t
            for a in r.answer_texts
            for t in normalized.get(r.contract_id, [])
        )
        for r in answerable
    )
    n = max(1, len(answerable))
    return {
        "answerable_questions": len(answerable),
        "literal_match": literal,
        "literal_match_share": round(literal / n, 4),
        "whitespace_normalized_match": norm,
        "whitespace_normalized_share": round(norm / n, 4),
        "undercounted_by_literal_rule": norm - literal,
    }


def build_quality_report(
    records: list[ContractQARecord],
    documents: dict[str, str],
    chunks: list[Chunk],
    chunk_size: int,
) -> dict:
    answerable = [r for r in records if not r.is_impossible]
    multi = sum(len(r.answer_texts) > 1 for r in answerable)
    per_contract = Counter(r.contract_id for r in records)
    per_category: dict[str, dict[str, int]] = {}
    for r in records:
        cat = per_category.setdefault(r.clause_category, {"answerable": 0, "total": 0})
        cat["total"] += 1
        cat["answerable"] += 0 if r.is_impossible else 1
    chunks_per_contract = Counter(c.contract_id for c in chunks)

    return {
        "contracts": len(documents),
        "question_records": len(records),
        "answerable": len(answerable),
        "unanswerable": len(records) - len(answerable),
        "unanswerable_share": round((len(records) - len(answerable)) / len(records), 4),
        "multi_answer_questions": multi,
        "multi_answer_share_of_answerable": round(multi / max(1, len(answerable)), 4),
        "questions_per_contract": _summary(list(per_contract.values())),
        "clause_categories": dict(sorted(per_category.items())),
        "contract_length_chars": _summary([len(t) for t in documents.values()]),
        "chunks": {
            "count": len(chunks),
            "length_chars": _summary([len(c.text) for c in chunks]),
            "per_contract": _summary(list(chunks_per_contract.values())),
        },
        "integrity": {
            "answer_span_offset_mismatches": count_span_mismatches(records),
            "duplicate_chunk_ids": len(chunks) - len({c.chunk_id for c in chunks}),
            "chunks_longer_than_chunk_size": sum(
                len(c.text) > chunk_size for c in chunks
            ),
            "empty_chunks": sum(not c.text.strip() for c in chunks),
        },
        "tiny_chunks": {
            "threshold_chars": TINY_CHUNK_CHARS,
            "count": sum(len(c.text) < TINY_CHUNK_CHARS for c in chunks),
            "share": round(
                sum(len(c.text) < TINY_CHUNK_CHARS for c in chunks) / len(chunks), 4
            ),
            "single_character": sum(len(c.text) == 1 for c in chunks),
        },
        "answer_coverage": answer_coverage(records, chunks),
    }


def export_processed(
    records: list[ContractQARecord], output_dir: Path, chunking: dict
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = build_contract_documents(records)
    title_by_id = {r.contract_id: r.contract_title for r in records}
    chunks = build_all_chunks(
        records,
        chunking["strategy"],
        chunking["chunk_size_chars"],
        chunking["chunk_overlap_chars"],
    )

    n_contracts = _write_jsonl(
        output_dir / "contracts.jsonl",
        (
            {
                "contract_id": cid,
                "contract_title": title_by_id[cid],
                "n_chars": len(text),
                "text": text,
            }
            for cid, text in documents.items()
        ),
    )
    n_qa = _write_jsonl(
        output_dir / "qa.jsonl",
        (
            {
                "contract_id": r.contract_id,
                "clause_category": r.clause_category,
                "question": r.question,
                "answer_texts": r.answer_texts,
                "answer_starts": r.answer_starts,
                "is_impossible": r.is_impossible,
            }
            for r in records
        ),
    )
    n_chunks = _write_jsonl(
        output_dir / "chunks.jsonl",
        (
            {
                "chunk_id": c.chunk_id,
                "contract_id": c.contract_id,
                "contract_title": c.contract_title,
                "position": c.position,
                "n_chars": len(c.text),
                "text": c.text,
            }
            for c in chunks
        ),
    )

    report = build_quality_report(
        records, documents, chunks, chunking["chunk_size_chars"]
    )
    report["chunking_config"] = dict(chunking)
    report["files"] = {
        "contracts.jsonl": n_contracts,
        "qa.jsonl": n_qa,
        "chunks.jsonl": n_chunks,
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Wrote %d contracts, %d questions, %d chunks to %s",
        n_contracts,
        n_qa,
        n_chunks,
        output_dir,
    )
    return report


def main() -> None:
    config = load_config()
    records = validate_records(
        load_raw_records(resolve_path(config["data"]["cuad_qa_path"]))
    )
    report = export_processed(
        records, resolve_path(config["data"]["processed_dir"]), config["chunking"]
    )
    keys = (
        "contracts",
        "question_records",
        "files",
        "integrity",
        "tiny_chunks",
        "answer_coverage",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
