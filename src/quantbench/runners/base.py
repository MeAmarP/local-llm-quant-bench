# Objective: Provide the abstract runner contract for all inference backends.

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """Abstract base class for backend runners."""

    @abstractmethod
    def run(self, prompt: str) -> dict:
        """Run one prompt and return raw run output."""
        raise NotImplementedError

