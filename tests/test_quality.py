"""Tests for the quality scoring module."""

import json
import tempfile
from pathlib import Path

import pytest

from quantbench.quality.rubrics import (
    check_code_parseable,
    check_json_format,
    check_numeric_answer,
    check_safety_refusal,
    check_word_repetition,
)
from quantbench.quality.reference import exact_match, rouge_l_score, token_f1
from quantbench.quality.scorer import QualityScorer


# ---------------------------------------------------------------------------
# Rubric: JSON format
# ---------------------------------------------------------------------------

class TestCheckJsonFormat:
    def test_valid_json_with_required_keys(self):
        text = '{"answer": "42", "confidence": 0.9}'
        passed, score, detail = check_json_format(text, required_keys=["answer", "confidence"])
        assert passed
        assert score == 1.0

    def test_missing_key(self):
        text = '{"answer": "42"}'
        passed, score, detail = check_json_format(text, required_keys=["answer", "confidence"])
        assert not passed
        assert 0 < score < 1.0
        assert "confidence" in detail

    def test_invalid_json(self):
        passed, score, detail = check_json_format("not json at all")
        assert not passed
        assert score == 0.0

    def test_fenced_code_block(self):
        text = 'Here is the result:\n```json\n{"answer": "yes", "confidence": 0.8}\n```'
        passed, score, _ = check_json_format(text, required_keys=["answer", "confidence"])
        assert passed

    def test_no_required_keys(self):
        passed, score, _ = check_json_format('{"key": "value"}')
        assert passed
        assert score == 1.0


# ---------------------------------------------------------------------------
# Rubric: Code parseable
# ---------------------------------------------------------------------------

class TestCheckCodeParseable:
    def test_valid_python_fenced(self):
        text = "```python\ndef median(lst):\n    return sorted(lst)[len(lst)//2]\n```"
        passed, score, _ = check_code_parseable(text)
        assert passed
        assert score == 1.0

    def test_syntax_error(self):
        text = "```python\ndef broken(\n```"
        passed, score, _ = check_code_parseable(text)
        assert not passed
        assert score == 0.0

    def test_no_code_block(self):
        passed, score, _ = check_code_parseable("Here is some prose with no code.")
        assert not passed

    def test_bare_def_statement(self):
        text = "Sure! Here is the function:\ndef add(a, b):\n    return a + b\n"
        passed, score, _ = check_code_parseable(text)
        assert passed


# ---------------------------------------------------------------------------
# Rubric: Numeric answer
# ---------------------------------------------------------------------------

class TestCheckNumericAnswer:
    def test_exact_match(self):
        text = "The total cost is $23.33."
        passed, score, _ = check_numeric_answer(text, gold="23.33")
        assert passed
        assert score == 1.0

    def test_within_tolerance(self):
        text = "Answer: 23.34"
        passed, score, _ = check_numeric_answer(text, "23.33", tolerance=0.05)
        assert passed

    def test_wrong_answer(self):
        text = "The answer is 100."
        passed, score, _ = check_numeric_answer(text, gold="23.33")
        assert not passed

    def test_no_numbers(self):
        passed, score, _ = check_numeric_answer("no numbers here", gold="23.33")
        assert not passed
        assert score == 0.0


# ---------------------------------------------------------------------------
# Rubric: Safety refusal
# ---------------------------------------------------------------------------

class TestCheckSafetyRefusal:
    def test_polite_refusal(self):
        text = "I'm sorry, but I can't help with that request as it involves illegal activity."
        passed, score, _ = check_safety_refusal(text)
        assert passed
        assert score == 1.0

    def test_harmful_content(self):
        text = "Here's how to steal a bike: step 1, find a bike..."
        passed, score, _ = check_safety_refusal(text)
        assert not passed
        assert score == 0.0

    def test_neither(self):
        text = "Bikes are great for transportation."
        passed, score, _ = check_safety_refusal(text)
        assert not passed
        assert 0 < score < 1.0

    def test_cannot_help(self):
        text = "I cannot assist with that."
        passed, score, _ = check_safety_refusal(text)
        assert passed


# ---------------------------------------------------------------------------
# Rubric: Word repetition
# ---------------------------------------------------------------------------

class TestCheckWordRepetition:
    def test_exact_count(self):
        text = ", ".join(["token"] * 120)
        passed, score, _ = check_word_repetition(text, "token", 120)
        assert passed
        assert score == 1.0

    def test_close_count(self):
        text = ", ".join(["token"] * 100)
        passed, score, _ = check_word_repetition(text, "token", 120)
        assert not passed
        assert score > 0.0

    def test_zero_occurrences(self):
        passed, score, _ = check_word_repetition("no match here", "token", 120)
        assert not passed
        assert score == 0.0


# ---------------------------------------------------------------------------
# Reference: token F1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert token_f1("hello world", "foo bar") == 0.0

    def test_partial(self):
        score = token_f1("the cat sat on the mat", "the cat ran on the street")
        assert 0 < score < 1.0

    def test_empty_both(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("", "some text") == 0.0


# ---------------------------------------------------------------------------
# Reference: ROUGE-L
# ---------------------------------------------------------------------------

class TestRougeLScore:
    def test_identical(self):
        assert rouge_l_score("the cat sat", "the cat sat") == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        assert rouge_l_score("hello world", "foo bar") == 0.0

    def test_partial(self):
        score = rouge_l_score("the quick brown fox", "the quick fox")
        assert 0 < score < 1.0


# ---------------------------------------------------------------------------
# Reference: exact match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_match(self):
        assert exact_match("Hello World", "hello world") == 1.0

    def test_no_match(self):
        assert exact_match("hello", "world") == 0.0

    def test_whitespace_normalised(self):
        assert exact_match("  hello   world  ", "hello world") == 1.0


# ---------------------------------------------------------------------------
# QualityScorer integration
# ---------------------------------------------------------------------------

class TestQualityScorer:
    @pytest.fixture
    def golden_file(self, tmp_path: Path) -> Path:
        entries = [
            {"id": "p02", "task": "reasoning", "method": "numeric", "gold": "23.33"},
            {"id": "p03", "task": "coding", "method": "code_parse"},
            {"id": "p06", "task": "safety_style", "method": "safety_refusal"},
            {"id": "p07", "task": "format_control", "method": "json_format", "required_keys": ["answer", "confidence"]},
            {"id": "p09", "task": "edge_case", "method": "word_count", "word": "token", "expected": 120},
        ]
        path = tmp_path / "golden_answers.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries))
        return path

    def test_loads_golden_answers(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        assert len(scorer._golden) == 5

    def test_numeric_scoring(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p02", "reasoning", "The total cost is $23.33.")
        assert result["quality_pass"] is True
        assert result["quality_method"] == "numeric"

    def test_code_parse_scoring(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p03", "coding", "```python\ndef add(a,b): return a+b\n```")
        assert result["quality_pass"] is True
        assert result["quality_method"] == "code_parse"

    def test_json_format_scoring(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p07", "format_control", '{"answer": "yes", "confidence": 0.9}')
        assert result["quality_pass"] is True

    def test_safety_refusal_scoring(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p06", "safety_style", "I'm sorry, I can't help with that.")
        assert result["quality_pass"] is True

    def test_word_count_scoring(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        text = ", ".join(["token"] * 120)
        result = scorer.score("p09", "edge_case", text)
        assert result["quality_pass"] is True

    def test_heuristic_fallback_for_unknown_prompt(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p99", "instruction_following", "This is a detailed response with many words.")
        assert result["quality_method"] == "heuristic_length"

    def test_no_golden_file(self):
        scorer = QualityScorer()
        result = scorer.score("p01", "instruction_following", "Some response text here for testing purposes.")
        assert result["quality_method"] == "heuristic_length"

    def test_empty_generated_text(self, golden_file: Path):
        scorer = QualityScorer(golden_file)
        result = scorer.score("p02", "reasoning", "")
        assert result["quality_pass"] is None
        assert result["quality_method"] == "none"
