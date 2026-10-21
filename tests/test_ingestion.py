from pathlib import Path

import pytest

from src.ingestion.load_cuad import load_raw_records
from src.ingestion.validate import DataValidationError, validate_records

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_cuad.json"


@pytest.fixture(scope="module")
def raw_records():
    return load_raw_records(FIXTURE_PATH)


def test_load_raw_records_count(raw_records):
    assert len(raw_records) == 8


def test_valid_fixture_passes_validation(raw_records):
    validated = validate_records(raw_records)
    assert len(validated) == 8
    assert all(hasattr(r, "contract_id") for r in validated)


def test_impossible_questions_have_no_answer_texts(raw_records):
    validated = validate_records(raw_records)
    impossible = [r for r in validated if r.is_impossible]
    assert len(impossible) == 2
    assert all(r.answer_texts == [] for r in impossible)


def test_answerable_questions_have_answer_texts(raw_records):
    validated = validate_records(raw_records)
    answerable = [r for r in validated if not r.is_impossible]
    assert len(answerable) == 6
    assert all(len(r.answer_texts) >= 1 for r in answerable)


def test_multi_answer_question_keeps_every_span(raw_records):

    validated = validate_records(raw_records)
    termination = [r for r in validated if r.clause_category == "Termination"][0]
    assert len(termination.answer_texts) == 2
    assert len(termination.answer_starts) == 2
    assert "thirty (30) days prior written notice" in termination.answer_texts
    assert "thirty (30) days notice" in termination.answer_texts


def test_missing_required_field_is_rejected(raw_records):
    broken = [dict(r) for r in raw_records]
    del broken[0]["question"]
    with pytest.raises(DataValidationError):
        validate_records(broken)


def test_impossible_with_answer_texts_is_rejected(raw_records):
    broken = [dict(r) for r in raw_records]
    broken[1]["is_impossible"] = True
    broken[1]["answer_texts"] = ["this should not be here"]
    broken[1]["answer_starts"] = [0]
    with pytest.raises(DataValidationError):
        validate_records(broken)


def test_negative_answer_start_is_rejected(raw_records):
    broken = [dict(r) for r in raw_records]
    broken[0]["answer_starts"] = [-5]
    with pytest.raises(DataValidationError):
        validate_records(broken)


def test_mismatched_answer_list_lengths_is_rejected(raw_records):

    broken = [dict(r) for r in raw_records]
    broken[0]["answer_texts"] = ["one", "two"]
    broken[0]["answer_starts"] = [5]
    with pytest.raises(DataValidationError):
        validate_records(broken)


def test_load_raw_records_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_records(Path("/nonexistent/path.json"))
