from pathlib import Path

def load_raw_text_data(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore")