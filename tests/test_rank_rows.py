"""Unit tests for the _rank_rows display-row builder extracted from the CLI commands."""

from skillrouter_test.cli import _rank_rows
from skillrouter_test.models import Skill


def _skills(*names: str) -> list[Skill]:
    return [Skill(name) for name in names]


def test_empty_ranking_yields_no_rows() -> None:
    assert _rank_rows([], _skills("a", "b")) == []


def test_single_entry_is_rank_one() -> None:
    pool = _skills("only")
    rows = _rank_rows([(0, 0.97)], pool)
    assert rows == [(1, "only", 0.97)]


def test_rank_is_one_indexed_and_preserves_input_order() -> None:
    # The helper must NOT re-sort; it renders the ranking in the order given.
    pool = _skills("alpha", "beta", "gamma")
    ranked = [(2, -1.5), (0, 0.8), (1, 0.1)]
    rows = _rank_rows(ranked, pool)
    assert rows == [(1, "gamma", -1.5), (2, "alpha", 0.8), (3, "beta", 0.1)]


def test_scores_passed_through_verbatim() -> None:
    pool = _skills("a", "b")
    rows = _rank_rows([(1, 12.0), (0, -0.25)], pool)
    assert [r[2] for r in rows] == [12.0, -0.25]
