import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP = "https://codeload.github.com/The-Atticus-Project/cuad/zip/refs/heads/main"
EXPECTED_CONTRACTS = 510
EXPECTED_QUESTIONS_PER_CONTRACT = 41
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "CUADv1.json"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def validate(path: Path) -> None:
    data = json.loads(path.read_text())["data"]
    if len(data) != EXPECTED_CONTRACTS:
        raise SystemExit(f"Expected {EXPECTED_CONTRACTS} contracts, found {len(data)}.")
    for contract in data:
        n = sum(len(p["qas"]) for p in contract["paragraphs"])
        if n != EXPECTED_QUESTIONS_PER_CONTRACT:
            raise SystemExit(f"{contract['title']}: expected 41 questions, found {n}.")


def main() -> None:
    if DEST.exists():
        print(f"{DEST} already exists; validating instead of re-downloading.")
        validate(DEST)
        print("OK")
        return

    print("Downloading CUAD repository archive...")
    outer = zipfile.ZipFile(io.BytesIO(fetch(REPO_ZIP)))
    inner_name = next(n for n in outer.namelist() if n.endswith("/data.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(inner.read("CUADv1.json"))
    validate(DEST)
    print(f"Wrote and validated {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
