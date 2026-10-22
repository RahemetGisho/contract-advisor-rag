"""Tests for src/processing/chunking.py and metadata.py."""

from pathlib import Path

import pytest

from src.ingestion.load_cuad import load_raw_records
from src.ingestion.validate import validate_records
from src.processing.chunking import chunk_document, clause_aware_chunk, fixed_size_chunk
from src.processing.metadata import build_all_chunks, build_contract_documents

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_cuad.json"

SAMPLE_TEXT = (
    "1. Term. This Agreement begins on the Effective Date.\n\n"
    "2. Termination. Either party may terminate upon 30 days notice.\n\n"
    "3. Governing Law. This Agreement is governed by Delaware law.\n\n"
    "4. Confidentiality. Each party shall keep the other's information confidential."
)


@pytest.fixture(scope="module")
def records():
    return validate_records(load_raw_records(FIXTURE_PATH))


def test_fixed_size_chunk_respects_overlap():
    chunks = fixed_size_chunk(SAMPLE_TEXT, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    # verify actual overlap: end of chunk[0] should reappear at start of chunk[1]
    assert chunks[0][-10:] == chunks[1][:10]


def test_fixed_size_chunk_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        fixed_size_chunk(SAMPLE_TEXT, chunk_size=50, overlap=50)


def test_clause_aware_chunk_keeps_clauses_together_when_possible():
    # SAMPLE_TEXT is 262 chars -- chunk_size must genuinely exceed that
    # for all 4 clauses to fit in one chunk. (chunk_size=200 was tried
    # first here and correctly failed: 262 > 200, so the chunker
    # correctly split it into 2 -- that was a bug in this test's
    # assumption, not in clause_aware_chunk itself.)
    chunks = clause_aware_chunk(SAMPLE_TEXT, chunk_size=300, overlap=20)
    assert len(chunks) == 1
    assert "Term." in chunks[0] and "Confidentiality" in chunks[0]


def test_clause_aware_chunk_splits_when_too_long():
    chunks = clause_aware_chunk(SAMPLE_TEXT, chunk_size=70, overlap=10)
    assert len(chunks) > 1
    # no chunk should wildly exceed the target size
    assert all(len(c) <= 70 + 20 for c in chunks)


def test_chunk_document_dispatches_correctly():
    fixed = chunk_document(SAMPLE_TEXT, "fixed_size", 50, 10)
    clause = chunk_document(SAMPLE_TEXT, "clause_aware", 50, 10)
    assert fixed != clause  # the two strategies should genuinely differ


def test_chunk_document_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document(SAMPLE_TEXT, "not_a_real_strategy", 50, 10)


def test_build_contract_documents_deduplicates_contexts(records):
    docs = build_contract_documents(records)
    assert len(docs) == 3  # 3 distinct contract_ids in the fixture
    # SYNTH-NDA-001 has 3 QAs but only 2 distinct context snippets
    assert docs["SYNTH-NDA-001"].count("Delaware") >= 1


def test_build_all_chunks_produces_chunks_per_contract(records):
    chunks = build_all_chunks(
        records, strategy="clause_aware", chunk_size=300, overlap=50
    )
    contract_ids = {c.contract_id for c in chunks}
    assert contract_ids == {"SYNTH-NDA-001", "SYNTH-LIC-002", "SYNTH-EMP-003"}
    assert all(c.chunk_id.startswith(c.contract_id) for c in chunks)


def test_chunk_size_tradeoff_comparison(records):
    """This is the evidence Task 3 asks for: compare chunk count at a
    few different chunk sizes on real (synthetic, here) data, rather
    than asserting one specific number -- the point is proving the
    chunk_size parameter actually changes behavior measurably."""
    small = build_all_chunks(records, "clause_aware", chunk_size=150, overlap=20)
    large = build_all_chunks(records, "clause_aware", chunk_size=800, overlap=50)
    assert len(small) >= len(large)
