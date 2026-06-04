"""Command-line harness for the SkillRouter pipeline.

Subcommands:
  retrieve  encoder-only ranking of a skill pool for one query
  rerank    cross-encoder reranking of a (small) candidate set
  route     full retrieve -> rerank pipeline
  demo      run the built-in sample pool through the full pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .models import Skill, SkillEncoder, SkillReranker, pick_device
from .sample_data import SAMPLE_QUERIES, SAMPLE_SKILLS

app = typer.Typer(add_completion=False, help="Test the SkillRouter retrieve-and-rerank models.")
console = Console()


def _load_skills(path: Path | None) -> list[Skill]:
    if path is None:
        return SAMPLE_SKILLS
    raw = json.loads(path.read_text())
    return [Skill(d["name"], d.get("description", ""), d.get("body", "")) for d in raw]


def _table(title: str, rows: list[tuple[int, str, float]]) -> Table:
    t = Table(title=title, show_lines=False)
    t.add_column("#", justify="right", style="dim")
    t.add_column("score", justify="right", style="cyan")
    t.add_column("skill", style="bold")
    for rank, name, score in rows:
        t.add_row(str(rank), f"{score:.4f}", name)
    return t


@app.command()
def retrieve(
    query: str,
    skills: Path = typer.Option(None, "--skills", "-s", help="JSON list of skills; omit for built-in pool."),
    top_k: int = typer.Option(20, "--top-k", "-k"),
    device: str = typer.Option("auto", "--device", "-d", help="auto|cuda|mps|cpu"),
) -> None:
    """Encoder-only retrieval: rank the whole pool by embedding similarity."""
    pool = _load_skills(skills)
    console.print(f"[dim]device={pick_device(device)}  pool={len(pool)}  loading encoder...[/dim]")
    enc = SkillEncoder(device=device)
    ranked = enc.rank(query, pool)[:top_k]
    rows = [(r + 1, pool[i].name, score) for r, (i, score) in enumerate(ranked)]
    console.print(_table(f"retrieve  ·  {query!r}", rows))


@app.command()
def rerank(
    query: str,
    skills: Path = typer.Option(None, "--skills", "-s", help="JSON list of candidate skills."),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Cross-encoder reranking of a candidate set (no first-stage retrieval)."""
    pool = _load_skills(skills)
    console.print(f"[dim]device={pick_device(device)}  candidates={len(pool)}  loading reranker...[/dim]")
    rk = SkillReranker(device=device)
    ranked = rk.rerank(query, pool)
    rows = [(r + 1, pool[i].name, score) for r, (i, score) in enumerate(ranked)]
    console.print(_table(f"rerank  ·  {query!r}", rows))


@app.command()
def route(
    query: str,
    skills: Path = typer.Option(None, "--skills", "-s"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Candidates kept after retrieval."),
    rerank_k: int = typer.Option(5, "--rerank-k", "-r", help="Final results shown."),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Full pipeline: encoder retrieves top-K, reranker re-scores them."""
    pool = _load_skills(skills)
    dev = pick_device(device)
    console.print(f"[dim]device={dev}  pool={len(pool)}  loading encoder + reranker...[/dim]")
    enc = SkillEncoder(device=device)
    rk = SkillReranker(device=device)

    retrieved = enc.rank(query, pool)[:top_k]
    cand = [pool[i] for i, _ in retrieved]
    reranked = rk.rerank(query, cand)[:rerank_k]
    rows = [(r + 1, cand[i].name, score) for r, (i, score) in enumerate(reranked)]
    console.print(_table(f"route  ·  {query!r}  (top-{top_k} -> rerank top-{rerank_k})", rows))


@app.command()
def demo(
    device: str = typer.Option("auto", "--device", "-d"),
    rerank_k: int = typer.Option(3, "--rerank-k", "-r"),
) -> None:
    """Run every built-in sample query through the full pipeline."""
    dev = pick_device(device)
    console.print(f"[dim]device={dev}  pool={len(SAMPLE_SKILLS)}  loading models once...[/dim]")
    enc = SkillEncoder(device=device)
    rk = SkillReranker(device=device)
    for q in SAMPLE_QUERIES:
        retrieved = enc.rank(q, SAMPLE_SKILLS)
        cand = [SAMPLE_SKILLS[i] for i, _ in retrieved]
        reranked = rk.rerank(q, cand)[:rerank_k]
        rows = [(r + 1, cand[i].name, score) for r, (i, score) in enumerate(reranked)]
        console.print(_table(q, rows))


if __name__ == "__main__":
    app()
