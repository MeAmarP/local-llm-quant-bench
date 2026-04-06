# Objective: Provide the abstract runner contract for all inference backends.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..models import PromptCase, RunResult, RunSpec


class BaseRunner(ABC):
    """
    Abstract base class for backend runners.

    Contract:
    - all backend runners accept a `RunSpec`
    - all backend runners consume a `PromptCase`
    - all backend runners return `RunResult`
    """

    def __init__(
        self,
        run_spec: RunSpec,
        *,
        generation_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_spec = run_spec
        self.generation_config = dict(generation_config or {})

    def load(self) -> None:
        """Optional model/backend loading hook."""
        return

    def unload(self) -> None:
        """Optional model/backend teardown hook."""
        return

    @abstractmethod
    def run_case(self, prompt_case: PromptCase) -> RunResult:
        """Run a structured prompt case and return a structured run result."""
        raise NotImplementedError

    def run(self, prompt: str, *, prompt_id: str = "adhoc", task: str = "adhoc") -> dict[str, Any]:
        """
        Compatibility wrapper for string-only call sites.

        Returns dict output so existing integrations that previously expected
        `run(prompt) -> dict` keep working.
        """
        case = PromptCase(id=prompt_id, task=task, prompt=prompt)
        return self.run_case(case).to_dict()
