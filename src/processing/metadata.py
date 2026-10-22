"""Build per-chunk records with metadata, from contract Q&A records.

build_contract_documents() reconstructs "the document" for a given
contract_id by deduplicating and concatenating the unique context
strings seen across that contract's Q&A records. Verified against the
real CUADv1.json release (all 510 contracts): every contract has
exactly one paragraph and one context string, shared identically
across all of that contract's questions -- so for the real dataset,
this reconstruction yields the EXACT original contract text, not an
approximation. The dedupe-and-join logic is still written defensively
to handle a hypothetical multi-context-per-contract release correctly
(CUAD's documented JSON schema technically allows multiple paragraphs
per contract, even though none of the 510 real ones use it), so this
code doesn't silently break if a future CUAD release or a different
dataset in this same format does have multiple distinct snippets per
contract -- it would then correctly concatenate them in first-seen
order, which is a reasonable approximation in that case, just not the
case this specific dataset exercises.
"""

from dataclasses import dataclass

from src.ingestion.schema import ContractQARecord
from src.processing.chunking import chunk_document


@dataclass
class Chunk:
    chunk_id: str
    contract_id: str
    contract_title: str
    text: str
    position: int


def build_contract_documents(records: list[ContractQARecord]) -> dict[str, str]:
    """Reconstruct the document for each contract_id by deduplicating and
    joining the unique context strings seen across that contract's Q&A
    records, in first-seen order. For the real CUAD release every contract
    has exactly one context string, so this returns the exact original
    contract text (see the module docstring)."""
    seen: dict[str, list[str]] = {}
    for r in records:
        seen.setdefault(r.contract_id, [])
        if r.context not in seen[r.contract_id]:
            seen[r.contract_id].append(r.context)
    return {contract_id: "\n\n".join(parts) for contract_id, parts in seen.items()}


def chunk_contract_full_text(
    contract_id: str,
    contract_title: str,
    full_text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk one contract's text (however it was sourced) into Chunk
    records with metadata. This is the function to call directly if
    you have real full contract text, bypassing build_contract_documents()'s
    QA-snippet approximation entirely."""
    pieces = chunk_document(full_text, strategy, chunk_size, overlap)
    return [
        Chunk(
            chunk_id=f"{contract_id}::chunk_{i}",
            contract_id=contract_id,
            contract_title=contract_title,
            text=piece,
            position=i,
        )
        for i, piece in enumerate(pieces)
    ]


def build_all_chunks(
    records: list[ContractQARecord], strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    documents = build_contract_documents(records)
    title_by_id = {r.contract_id: r.contract_title for r in records}

    all_chunks: list[Chunk] = []
    for contract_id, text in documents.items():
        all_chunks.extend(
            chunk_contract_full_text(
                contract_id,
                title_by_id[contract_id],
                text,
                strategy,
                chunk_size,
                overlap,
            )
        )
    return all_chunks
