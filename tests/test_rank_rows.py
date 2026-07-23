"""Contract tests for the ``_rank_rows`` display-row builder.

Contract (from cli.py):
  _rank_rows(ranked, skills) -> [(rank, skills[idx].name, score), ...]
    - rank is 1-indexed, assigned by position in ``ranked`` (enumerate order)
    - the input order is preserved verbatim — it is NOT re-sorted by score
    - score is passed through unchanged (no rounding/normalization)
    - ``skills`` is indexed by the first element of each (idx, score) tuple;
      an out-of-range idx raises IndexError
    - deterministic + idempotent: repeated calls with the same input are equal
"""

import pytest

from skillrouter_test.cli import _rank_rows
from skillrouter_test.models import Skill


def _skills(*names: str) -> list[Skill]:
    return [Skill(name) for name in names]


# --- boundaries ------------------------------------------------------------

def test_empty_ranking_yields_no_rows() -> None:
    assert _rank_rows([], _skills("a", "b")) == []


def test_single_entry_is_rank_one() -> None:
    pool = _skills("only")
    assert _rank_rows([(0, 0.97)], pool) == [(1, "only", 0.97)]


def test_rank_is_one_indexed_and_preserves_input_order() -> None:
    # The helper must NOT re-sort; it renders the ranking in the order given.
    pool = _skills("alpha", "beta", "gamma")
    ranked = [(2, -1.5), (0, 0.8), (1, 0.1)]
    assert _rank_rows(ranked, pool) == [
        (1, "gamma", -1.5),
        (2, "alpha", 0.8),
        (3, "beta", 0.1),
    ]


# --- rank numbering is a contiguous 1..N sequence -------------------------

def test_ranks_are_contiguous_one_to_n() -> None:
    pool = _skills("a", "b", "c", "d")
    ranked = [(3, 0.0), (0, 0.0), (2, 0.0), (1, 0.0)]
    rows = _rank_rows(ranked, pool)
    assert [r[0] for r in rows] == [1, 2, 3, 4]


def test_input_with_tied_scores_keeps_input_order() -> None:
    # Determinism tie-break: equal scores do not perturb the given order.
    pool = _skills("x", "y", "z")
    ranked = [(1, 0.5), (2, 0.5), (0, 0.5)]
    assert _rank_rows(ranked, pool) == [(1, "y", 0.5), (2, "z", 0.5), (3, "x", 0.5)]


# --- score pass-through (no mutation of values) ---------------------------

def test_scores_passed_through_verbatim() -> None:
    pool = _skills("a", "b")
    rows = _rank_rows([(1, 12.0), (0, -0.25)], pool)
    assert [r[2] for r in rows] == [12.0, -0.25]


@pytest.mark.parametrize(
    "score",
    [0.0, -0.0, 1e-30, -1e30, 0.5, float("inf"), float("-inf")],
)
def test_extreme_and_special_floats_pass_through_unchanged(score: float) -> None:
    pool = _skills("a")
    assert _rank_rows([(0, score)], pool)[0][2] == score


def test_nan_score_is_preserved_as_nan() -> None:
    # NaN is the one float where == is False; assert isnan explicitly.
    pool = _skills("a")
    row = _rank_rows([(0, float("nan"))], pool)[0]
    import math
    assert math.isnan(row[2])


# --- name lookup uses the index, in order ---------------------------------

def test_each_row_name_comes_from_skills_at_the_ranked_index() -> None:
    pool = _skills("zero", "one", "two")
    ranked = [(2, 0.9), (0, 0.5), (1, 0.1)]
    assert [r[1] for r in _rank_rows(ranked, pool)] == ["two", "zero", "one"]


# --- determinism / idempotency (L4) ---------------------------------------

def test_deterministic_across_repeated_calls() -> None:
    pool = _skills("a", "b", "c")
    ranked = [(1, 0.3), (0, 0.2), (2, 0.1)]
    first = _rank_rows(ranked, pool)
    for _ in range(5):
        assert _rank_rows(ranked, pool) == first


def test_idempotent_under_repeated_index() -> None:
    # The contract indexes ``skills`` by the given idx without deduping, so a
    # repeated index yields a repeated skill name with advancing rank.
    pool = _skills("only")
    rows = _rank_rows([(0, 0.7), (0, 0.6), (0, 0.5)], pool)
    assert rows == [(1, "only", 0.7), (2, "only", 0.6), (3, "only", 0.5)]


# --- error semantics ------------------------------------------------------

def test_out_of_range_index_raises_index_error() -> None:
    pool = _skills("a", "b")
    with pytest.raises(IndexError):
        _rank_rows([(5, 0.1)], pool)
