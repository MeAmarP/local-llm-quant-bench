"""QualityScorer: dispatches prompt responses to rubric or reference checks.

Loading golden answers is optional — if no golden answer file is provided or
a prompt has no entry in the file, task-based heuristics are used instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .rubrics import (
    check_code_parseable,
    check_json_format,
    check_numeric_answer,
    check_safety_refusal,
    check_word_repetition,
)
from .reference import rouge_l_score, token_f1

logger = logging.getLogger(__name__)


# Heuristic task → method mapping for prompts that have no golden answer entry
_TASK_HEURISTICS: dict[str, str] = {
    "format_control": "json_format",
    "coding": "code_parse",
    "safety_style": "safety_refusal",
    "reasoning": "heuristic_length",
    "summarization": "heuristic_length",
    "instruction_following": "heuristic_length",
    "factual": "heuristic_length",
    "edge_case": "heuristic_length",
    "app_realism": "heuristic_length",
    "long_context_retrieval": "heuristic_length",
}


class QualityScorer:
    """Score generated responses using rubrics and optional reference answers.

    Args:
        golden_answers_path: Path to a JSONL file where each line has:
            - ``id``: prompt ID matching PromptCase.id
            - ``method``: one of ``numeric``, ``code_parse``, ``json_format``,
              ``safety_refusal``, ``word_count``, ``rouge_l``, ``token_f1``
            - Additional method-specific fields (``gold``, ``required_keys``,
              ``word``, ``expected``)
    """

    def __init__(self, golden_answers_path: str | Path | None = None) -> None:
        self._golden: dict[str, dict[str, Any]] = {}
        if golden_answers_path:
            self._load_golden(Path(golden_answers_path))

    def _load_golden(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"QualityScorer: golden answers file not found: {path}")
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    prompt_id = rec.get("id")
                    if prompt_id:
                        self._golden[prompt_id] = rec
                except json.JSONDecodeError as exc:
                    logger.warning(f"QualityScorer: skipping malformed line: {exc}")
        logger.info(f"QualityScorer: loaded {len(self._golden)} golden answer entries from {path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, prompt_id: str, task: str, generated_text: str) -> dict[str, Any]:
        """Score a generated response.

        Returns:
            Dict with keys: ``quality_pass``, ``quality_score``,
            ``quality_method``, ``quality_details``.
        """
        golden = self._golden.get(prompt_id)
        method = golden.get("method") if golden else _TASK_HEURISTICS.get(task)

        if not generated_text or not generated_text.strip():
            return _empty_result("no generated text")

        try:
            return self._dispatch(method, generated_text, golden)
        except Exception as exc:
            logger.warning(f"QualityScorer.score failed for {prompt_id}: {exc}")
            return _empty_result(f"scorer error: {exc}")

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        method: str | None,
        text: str,
        golden: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if method == "json_format":
            required_keys = (golden or {}).get("required_keys")
            passed, score, detail = check_json_format(text, required_keys)
            return _result(passed, score, "json_format", detail)

        if method == "code_parse":
            passed, score, detail = check_code_parseable(text)
            return _result(passed, score, "code_parse", detail)

        if method == "numeric":
            gold = (golden or {}).get("gold")
            if gold is None:
                return _empty_result("numeric method requires 'gold' in golden answer")
            passed, score, detail = check_numeric_answer(text, gold)
            return _result(passed, score, "numeric", detail)

        if method == "safety_refusal":
            passed, score, detail = check_safety_refusal(text)
            return _result(passed, score, "safety_refusal", detail)

        if method == "word_count":
            g = golden or {}
            word = g.get("word")
            expected = g.get("expected")
            if word is None or expected is None:
                return _empty_result("word_count method requires 'word' and 'expected'")
            passed, score, detail = check_word_repetition(text, word, int(expected))
            return _result(passed, score, "word_count", detail)

        if method == "rouge_l":
            ref = (golden or {}).get("gold", "")
            score = rouge_l_score(text, ref)
            return _result(score >= 0.3, score, "rouge_l", f"ROUGE-L F1: {score:.4f}")

        if method == "token_f1":
            ref = (golden or {}).get("gold", "")
            score = token_f1(text, ref)
            return _result(score >= 0.4, score, "token_f1", f"Token F1: {score:.4f}")

        if method and method.startswith("heuristic"):
            # Generic length heuristic: non-empty response of at least 10 words
            word_count = len(text.split())
            passed = word_count >= 10
            score = min(1.0, word_count / 50)
            return _result(passed, score, "heuristic_length", f"word count: {word_count}")

        return _empty_result(f"no scoring method available (task heuristic: {method})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    passed: bool,
    score: float,
    method: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "quality_pass": passed,
        "quality_score": round(float(score), 4),
        "quality_method": method,
        "quality_details": {"detail": detail},
    }


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "quality_pass": None,
        "quality_score": None,
        "quality_method": "none",
        "quality_details": {"detail": reason},
    }
