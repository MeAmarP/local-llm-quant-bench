# Objective: Load prompts from JSONL files and prompt-set directories.

import json
from pathlib import Path

from .models import PromptCase


def load_prompts(path: str | Path) -> list[PromptCase]:
    """Load prompts from a JSONL file or directory of JSONL files.
    
    Args:
        path: Path to a .jsonl file or directory containing .jsonl files
        
    Returns:
        List of PromptCase objects
        
    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If no JSONL files found in directory
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    prompts: list[PromptCase] = []
    
    if path.is_file():
        # Load from single JSONL file
        if path.suffix != ".jsonl":
            raise ValueError(f"Expected .jsonl file, got: {path}")
        prompts.extend(_load_jsonl_file(path))
    else:
        # Load from all JSONL files in directory (recursively)
        jsonl_files = sorted(path.rglob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in directory: {path}")
        for jsonl_file in jsonl_files:
            prompts.extend(_load_jsonl_file(jsonl_file))
    
    return prompts


def _load_jsonl_file(path: Path) -> list[PromptCase]:
    """Load prompts from a single JSONL file.
    
    Args:
        path: Path to .jsonl file
        
    Returns:
        List of PromptCase objects
        
    Raises:
        json.JSONDecodeError: If line is not valid JSON
        ValueError: If JSON object does not match PromptCase schema
    """
    prompts: list[PromptCase] = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                prompt = PromptCase(**data)
                prompts.append(prompt)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at {path}:{line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                )
            except TypeError as e:
                raise ValueError(
                    f"Invalid prompt schema at {path}:{line_num}: {e}"
                )
    
    return prompts

