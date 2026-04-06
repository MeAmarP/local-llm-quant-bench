# Objective: Implement a Hugging Face Transformers backend runner.

from __future__ import annotations

from ..models import PromptCase, RunResult
from .base import BaseRunner


class TransformersRunner(BaseRunner):
    """Placeholder Transformers runner."""

    def run_case(self, prompt_case: PromptCase) -> RunResult:
        del prompt_case
        raise NotImplementedError("TransformersRunner is not implemented yet.")
