# Objective: Write benchmark artifacts (JSONL/JSON/Markdown) to the results directory.

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return resolved path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

