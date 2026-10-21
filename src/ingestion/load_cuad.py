import json
from pathlib import Path

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def download_cuad(dest_path: Path) -> Path:

    from datasets import load_dataset

    logger.info("Downloading CUAD from Hugging Face (theatticusproject/cuad-qa)...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train")

    # The HF `datasets` version is already flattened to one row per
    # QA pair (unlike the raw SQuAD-nested release) -- normalize field
    # names to what parse_hf_format() expects below.
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(dest_path))
    logger.info("Saved %d records to %s", len(ds), dest_path)
    return dest_path


def load_local_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No CUAD data file at {path}. Run download_cuad() first, "
            "or point this at a local file (e.g. the synthetic test fixture)."
        )
    with open(path) as f:
        content = f.read()

    # Support both a plain JSON array and JSON-lines (what datasets.to_json produces)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def parse_squad_format(raw_data: dict) -> list[dict]:

    records: list[dict] = []
    for entry in raw_data.get("data", []):
        title = entry.get("title", "unknown")
        for paragraph in entry.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id", "")
                clause_category = qa_id.split("__")[-1] if "__" in qa_id else "unknown"
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", len(answers) == 0)
                # Keep EVERY answer span, not just the first: 38.9% of CUAD's
                # answerable questions have more than one valid span (see
                # schema.py's docstring) -- dropping the rest would silently
                # understate retrieval recall for those questions.
                records.append(
                    {
                        "contract_id": title,
                        "contract_title": title,
                        "clause_category": clause_category,
                        "question": qa.get("question", ""),
                        "context": context,
                        "answer_texts": [a["text"] for a in answers],
                        "answer_starts": [a["answer_start"] for a in answers],
                        "is_impossible": is_impossible,
                    }
                )
    return records


def load_raw_records(path: Path) -> list[dict]:

    raw = load_local_json(path)
    if isinstance(raw, dict) and "data" in raw:
        return parse_squad_format(raw)
    if isinstance(raw, list):
        return raw
    raise ValueError(
        f"Unrecognized data format at {path}: expected a list or a SQuAD-style dict"
    )
