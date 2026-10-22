import re

_CLAUSE_BOUNDARY = re.compile(r"\n\s*\n|(?=\n\s*(?:\d+\.|\(\w+\)|[A-Z][A-Z \t]{3,}:))")


def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def clause_aware_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in _CLAUSE_BOUNDARY.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # carry the tail of the previous chunk forward as overlap
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            # a single paragraph longer than chunk_size on its own --
            # fall back to fixed-size splitting for just this paragraph
            # rather than producing one giant oversized chunk.
            chunks.extend(fixed_size_chunk(para, chunk_size, overlap))
            current = ""

        if len(current) > chunk_size:
            chunks.extend(fixed_size_chunk(current, chunk_size, overlap))
            current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    text: str, strategy: str, chunk_size: int, overlap: int
) -> list[str]:
    if strategy == "fixed_size":
        return fixed_size_chunk(text, chunk_size, overlap)
    if strategy == "clause_aware":
        return clause_aware_chunk(text, chunk_size, overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
