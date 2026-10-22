import json
from pathlib import Path

import pytest

from src.ingestion.load_cuad import load_raw_records
from src.ingestion.schema import ContractQARecord
from src.ingestion.validate import validate_records
from src.processing.export_processed import (
    answer_coverage,
    count_span_mismatches,
    export_processed,
)
from src.processing.metadata import Chunk, build_all_chunks

FIXTURE = Path(__file__).parent / "fixtures" / "sample_cuad.json"
CHUNKING = {
    "strategy": "fixed_size",
    "chunk_size_chars": 200,
    "chunk_overlap_chars": 40,
}


@pytest.fixture(scope="module")
def records():
    return validate_records(load_raw_records(FIXTURE))


@pytest.fixture
def exported(records, tmp_path):
    out = tmp_path / "processed"
    return out, export_processed(records, out, CHUNKING)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_four_files_are_written_with_the_right_row_counts(exported, records):
    out, report = exported
    expected_chunks = build_all_chunks(
        records,
        CHUNKING["strategy"],
        CHUNKING["chunk_size_chars"],
        CHUNKING["chunk_overlap_chars"],
    )
    assert len(_rows(out / "contracts.jsonl")) == 3
    assert len(_rows(out / "qa.jsonl")) == 8
    assert len(_rows(out / "chunks.jsonl")) == len(expected_chunks)
    assert (out / "data_quality_report.json").exists()
    assert report["files"] == {
        "contracts.jsonl": 3,
        "qa.jsonl": 8,
        "chunks.jsonl": len(expected_chunks),
    }


def test_question_rows_never_repeat_the_contract_text(exported):

    out, _ = exported
    for row in _rows(out / "qa.jsonl"):
        assert "context" not in row and "text" not in row
        assert set(row) == {
            "contract_id",
            "clause_category",
            "question",
            "answer_texts",
            "answer_starts",
            "is_impossible",
        }


def test_a_single_context_contract_is_saved_exactly_as_is(tmp_path):

    text = "Section 1. Term.\n\nThis Agreement lasts one year."
    records = [
        _record(text, "one year", text.find("one year")),
        _record(text, "Section 1", 0),
    ]
    export_processed(records, tmp_path, CHUNKING)
    (row,) = _rows(tmp_path / "contracts.jsonl")
    assert row["text"] == text and row["n_chars"] == len(text)


def test_a_multi_context_contract_joins_each_unique_snippet_once_in_order(
    exported, records
):

    out, _ = exported
    nda = next(
        r for r in _rows(out / "contracts.jsonl") if r["contract_id"] == "SYNTH-NDA-001"
    )
    snippets = list(
        dict.fromkeys(r.context for r in records if r.contract_id == "SYNTH-NDA-001")
    )
    assert len(snippets) > 1
    assert all(nda["text"].count(snip) == 1 for snip in snippets)
    assert [nda["text"].index(snip) for snip in snippets] == sorted(
        nda["text"].index(snip) for snip in snippets
    )


def test_chunk_ids_are_unique_and_match_what_the_index_uses(exported, records):
    out, _ = exported
    exported_ids = [r["chunk_id"] for r in _rows(out / "chunks.jsonl")]
    index_chunks = build_all_chunks(
        records,
        CHUNKING["strategy"],
        CHUNKING["chunk_size_chars"],
        CHUNKING["chunk_overlap_chars"],
    )
    index_ids = [c.chunk_id for c in index_chunks]
    assert exported_ids == index_ids
    assert len(set(exported_ids)) == len(exported_ids)


def test_every_chunk_is_text_that_really_occurs_in_its_contract(exported):
    out, _ = exported
    contracts = {r["contract_id"]: r["text"] for r in _rows(out / "contracts.jsonl")}
    for chunk in _rows(out / "chunks.jsonl"):
        assert chunk["text"] in contracts[chunk["contract_id"]]
        assert chunk["n_chars"] == len(chunk["text"])


def test_report_counts_match_the_fixture(exported):
    _, report = exported
    assert report["contracts"] == 3
    assert report["question_records"] == 8
    assert report["answerable"] == 6 and report["unanswerable"] == 2
    assert report["multi_answer_questions"] == 1  # the Termination question
    assert report["clause_categories"]["Termination"] == {"answerable": 1, "total": 1}
    assert report["integrity"]["duplicate_chunk_ids"] == 0
    assert report["integrity"]["empty_chunks"] == 0


def test_report_on_disk_equals_the_returned_report(exported):
    out, report = exported
    assert (
        json.loads((out / "data_quality_report.json").read_text(encoding="utf-8"))
        == report
    )


def _record(context: str, answer: str, start: int) -> ContractQARecord:
    return ContractQARecord(
        contract_id="C",
        contract_title="T",
        clause_category="X",
        question="q?",
        context=context,
        answer_texts=[answer],
        answer_starts=[start],
    )


def test_span_check_passes_when_the_offset_is_right():
    context = "Alpha beta gamma delta"
    assert (
        count_span_mismatches(
            [_record(context, "beta gamma", context.find("beta gamma"))]
        )
        == 0
    )


def test_span_check_catches_a_wrong_offset():
    assert (
        count_span_mismatches([_record("Alpha beta gamma delta", "beta gamma", 0)]) == 1
    )


def test_rerunning_overwrites_cleanly_and_leaves_no_temp_files(records, tmp_path):
    out = tmp_path / "processed"
    first = export_processed(records, out, CHUNKING)
    second = export_processed(records, out, CHUNKING)
    assert first == second
    assert not list(out.glob("*.tmp"))


def test_missing_output_directories_are_created(records, tmp_path):
    out = tmp_path / "a" / "b" / "processed"
    export_processed(records, out, CHUNKING)
    assert (out / "chunks.jsonl").exists()


def _chunk(text: str, contract_id: str = "C") -> Chunk:
    return Chunk(
        chunk_id=f"{contract_id}::{text[:5]}",
        contract_id=contract_id,
        contract_title="T",
        text=text,
        position=0,
    )


def test_coverage_finds_an_answer_that_sits_in_a_chunk():
    rec = _record("the term is one year", "one year", 12)
    cov = answer_coverage([rec], [_chunk("the term is one year")])
    assert cov["literal_match"] == 1 and cov["literal_match_share"] == 1.0


def test_coverage_counts_an_answer_no_chunk_contains():

    rec = _record("one year term", "one year term", 0)
    cov = answer_coverage([rec], [_chunk("one year"), _chunk("year term")])
    assert cov["literal_match"] == 0 and cov["whitespace_normalized_match"] == 0


def test_whitespace_only_difference_is_literal_miss_but_normalized_hit():

    rec = _record("alpha    beta", "alpha    beta", 0)
    cov = answer_coverage([rec], [_chunk("alpha beta")])
    assert cov["literal_match"] == 0
    assert cov["whitespace_normalized_match"] == 1
    assert cov["undercounted_by_literal_rule"] == 1


def test_coverage_only_looks_in_the_questions_own_contract():
    rec = _record("answer here", "answer here", 0)  # contract "C"
    cov = answer_coverage([rec], [_chunk("answer here", contract_id="OTHER")])
    assert cov["literal_match"] == 0


def test_unanswerable_questions_are_excluded_from_coverage():
    rec = ContractQARecord(
        contract_id="C",
        contract_title="T",
        clause_category="X",
        question="q?",
        context="ctx",
        is_impossible=True,
    )
    assert answer_coverage([rec], [_chunk("anything")])["answerable_questions"] == 0


def test_report_counts_tiny_chunks(tmp_path):
    text = "A. " + "x" * 150
    records = [_record(text, "xxx", 3)]
    small_chunks = {
        "strategy": "fixed_size",
        "chunk_size_chars": 100,
        "chunk_overlap_chars": 10,
    }
    report = export_processed(records, tmp_path, small_chunks)
    assert report["tiny_chunks"]["threshold_chars"] == 50
    assert report["tiny_chunks"]["count"] == sum(
        r["n_chars"] < 50 for r in _rows(tmp_path / "chunks.jsonl")
    )
    assert "answer_coverage" in report
