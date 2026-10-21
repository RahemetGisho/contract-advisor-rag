from pydantic import ValidationError

from src.ingestion.schema import ContractQARecord
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when one or more raw records fail schema validation."""


def validate_records(raw_records: list[dict]) -> list[ContractQARecord]:
    validated: list[ContractQARecord] = []
    failures: list[dict] = []

    for i, raw in enumerate(raw_records):
        try:
            record = ContractQARecord(**raw)
        except ValidationError as exc:
            failures.append({"index": i, "errors": exc.errors(), "raw": raw})
            continue

        # Cross-field rule that pydantic's per-field validators can't
        # express cleanly: an "impossible" question must carry no
        # answer spans, and an answerable one must carry at least one --
        # these are both real data-integrity bugs if violated, not just
        # style issues.
        if record.is_impossible and record.answer_texts:
            failures.append(
                {
                    "index": i,
                    "errors": "is_impossible=True but answer_texts is non-empty",
                    "raw": raw,
                }
            )
            continue
        if not record.is_impossible and not record.answer_texts:
            failures.append(
                {
                    "index": i,
                    "errors": "is_impossible=False but answer_texts is empty",
                    "raw": raw,
                }
            )
            continue

        validated.append(record)

    if failures:
        logger.error(
            "Validation failed for %d/%d records", len(failures), len(raw_records)
        )
        raise DataValidationError(
            f"{len(failures)} of {len(raw_records)} records failed validation. "
            f"First failure: {failures[0]}"
        )

    logger.info("Validated %d contract Q&A records", len(validated))
    return validated
