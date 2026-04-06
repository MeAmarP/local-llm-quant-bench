# Objective: Implement a llama.cpp backend runner.

from quantbench.runners.base import BaseRunner


class LlamaCppRunner(BaseRunner):
    """Placeholder llama.cpp runner."""

    def run(self, prompt: str) -> dict:
        raise NotImplementedError("LlamaCppRunner is not implemented yet.")

