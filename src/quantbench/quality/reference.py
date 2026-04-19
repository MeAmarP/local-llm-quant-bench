"""Reference-based scoring utilities (token F1, ROUGE-L, exact match).

Tries to use the ``rouge_score`` library when available; falls back to a
pure-Python token F1 implementation that requires no dependencies.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase-normalize and split text into word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def exact_match(pred: str, ref: str) -> float:
    """Return 1.0 if pred == ref after normalising whitespace, else 0.0."""
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    return 1.0 if norm(pred) == norm(ref) else 0.0


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def token_f1(pred: str, ref: str) -> float:
    """Compute token-level F1 between prediction and reference.

    Uses multi-set overlap (counts matter) to measure coverage.
    Returns float in [0, 1].
    """
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    from collections import Counter
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence using DP (O(n*m))."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Space-optimised two-row DP
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def rouge_l_score(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference.

    Tries the ``rouge_score`` library first; falls back to a pure-Python LCS
    implementation that produces equivalent results.

    Returns float in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore[import]
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(ref, pred)
        return result["rougeL"].fmeasure
    except ImportError:
        pass

    # Pure-Python fallback via LCS
    pred_tokens = _tokenize(pred)
    ref_tokens = _tokenize(ref)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
