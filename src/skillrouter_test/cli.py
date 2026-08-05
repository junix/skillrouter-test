"""Command-line harness for the SkillRouter pipeline.

Subcommands:
  retrieve  encoder-only ranking of a skill pool for one query
  rerank    cross-encoder reranking of a (small) candidate set
  route     full retrieve -> rerank pipeline
  demo      run the built-in sample pool through the full pipeline
  batch     route many queries with a single model load

Weight-lifecycle subcommands (skill design-cli §十九; weights live in
:mod:`skillrouter_test.weights`):
  available   read-only probe: are both checkpoints cached? (no model load, no network)
  download    idempotently fetch the checkpoints (+ --force / --dry-run / --json)
  describe    self-describe: capabilities, managed models, source path
  doctor      report weight readiness (missing weights are NOT fatal)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import weights as W
from .sample_data import SAMPLE_QUERIES, SAMPLE_SKILLS


def _source_path() -> str:
    here = Path(__file__).resolve()
    root = next((p for p in [here, *here.parents] if (p / "pyproject.toml").exists()), here)
    return str(root).replace(os.path.expanduser("~"), "~")


app = typer.Typer(
    add_completion=False,
    help="Test the SkillRouter retrieve-and-rerank models.",
    epilog=f"Source: {_source_path()}",
)
console = Console()


def _load_skills(path: Path | None):
    from .models import Skill  # deferred: keep `available`/`doctor` torch-free

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


def _rank_rows(ranked: list[tuple[int, float]], skills) -> list[tuple[int, str, float]]:
    """Build 1-indexed (rank, skill_name, score) display rows from a (index, score) ranking."""
    return [(r + 1, skills[i].name, score) for r, (i, score) in enumerate(ranked)]


@app.command()
def retrieve(
    query: str,
    skills: Path = typer.Option(None, "--skills", "-s", help="JSON list of skills; omit for built-in pool."),
    top_k: int = typer.Option(20, "--top-k", "-k"),
    device: str = typer.Option("auto", "--device", "-d", help="auto|cuda|mps|cpu"),
) -> None:
    """Encoder-only retrieval: rank the whole pool by embedding similarity."""
    from .models import SkillEncoder, pick_device

    pool = _load_skills(skills)
    console.print(f"[dim]device={pick_device(device)}  pool={len(pool)}  loading encoder...[/dim]")
    enc = SkillEncoder(device=device)
    ranked = enc.rank(query, pool)[:top_k]
    rows = _rank_rows(ranked, pool)
    console.print(_table(f"retrieve  ·  {query!r}", rows))


@app.command()
def rerank(
    query: str,
    skills: Path = typer.Option(None, "--skills", "-s", help="JSON list of candidate skills."),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Cross-encoder reranking of a candidate set (no first-stage retrieval)."""
    from .models import SkillReranker, pick_device

    pool = _load_skills(skills)
    console.print(f"[dim]device={pick_device(device)}  candidates={len(pool)}  loading reranker...[/dim]")
    rk = SkillReranker(device=device)
    ranked = rk.rerank(query, pool)
    rows = _rank_rows(ranked, pool)
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
    from .models import SkillEncoder, SkillReranker, pick_device

    pool = _load_skills(skills)
    dev = pick_device(device)
    console.print(f"[dim]device={dev}  pool={len(pool)}  loading encoder + reranker...[/dim]")
    enc = SkillEncoder(device=device)
    rk = SkillReranker(device=device)

    retrieved = enc.rank(query, pool)[:top_k]
    cand = [pool[i] for i, _ in retrieved]
    reranked = rk.rerank(query, cand)[:rerank_k]
    rows = _rank_rows(reranked, cand)
    console.print(_table(f"route  ·  {query!r}  (top-{top_k} -> rerank top-{rerank_k})", rows))


@app.command()
def demo(
    device: str = typer.Option("auto", "--device", "-d"),
    rerank_k: int = typer.Option(3, "--rerank-k", "-r"),
) -> None:
    """Run every built-in sample query through the full pipeline."""
    from .models import SkillEncoder, SkillReranker, pick_device

    dev = pick_device(device)
    console.print(f"[dim]device={dev}  pool={len(SAMPLE_SKILLS)}  loading models once...[/dim]")
    enc = SkillEncoder(device=device)
    rk = SkillReranker(device=device)
    for q in SAMPLE_QUERIES:
        retrieved = enc.rank(q, SAMPLE_SKILLS)
        cand = [SAMPLE_SKILLS[i] for i, _ in retrieved]
        reranked = rk.rerank(q, cand)[:rerank_k]
        rows = _rank_rows(reranked, cand)
        console.print(_table(q, rows))


@app.command()
def batch(
    queries: Path = typer.Argument(..., help="Text file: one query per line (optionally `query\\texpected_skill`)."),
    skills: Path = typer.Option(None, "--skills", "-s"),
    top_k: int = typer.Option(20, "--top-k", "-k"),
    rerank_k: int = typer.Option(5, "--rerank-k", "-r"),
    device: str = typer.Option("auto", "--device", "-d"),
) -> None:
    """Route many queries with a single model load; report Hit@1 if expectations given."""
    from .models import SkillEncoder, SkillReranker, pick_device

    pool = _load_skills(skills)
    dev = pick_device(device)
    console.print(f"[dim]device={dev}  pool={len(pool)}  loading models once...[/dim]")
    enc = SkillEncoder(device=device)
    rk = SkillReranker(device=device)

    hits = total = 0
    for line in queries.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q, _, expected = line.partition("\t")
        q, expected = q.strip(), expected.strip()
        retrieved = enc.rank(q, pool)[:top_k]
        cand = [pool[i] for i, _ in retrieved]
        reranked = rk.rerank(q, cand)[:rerank_k]
        rows = _rank_rows(reranked, cand)
        title = q if not expected else f"{q}   [expect: {expected}]"
        if expected:
            total += 1
            top1 = cand[reranked[0][0]].name
            ok = top1 == expected
            hits += ok
            title = f"[{'green' if ok else 'red'}]{'✓' if ok else '✗'}[/] " + title
        console.print(_table(title, rows))
    if total:
        console.print(f"\n[bold]Hit@1: {hits}/{total} = {hits / total:.0%}[/bold]")


# --- weight-lifecycle subcommands (design-cli §十九) ------------------------

def _status_payload() -> dict:
    statuses = W.probe_all()
    models = [s.to_ready_dict() if s.ready else s.to_missing_dict() for s in statuses]
    return {"ok": all(s.ready for s in statuses), "models": models}


@app.command()
def available(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (stable contract)."),
) -> None:
    """Read-only probe: are both SkillRouter checkpoints cached?

    Never loads a model, never hits the network, never writes. Missing weights
    are reported, not fatal (always exits 0)."""
    payload = _status_payload()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    table = Table("model", "repo", "status")
    for m in payload["models"]:
        table.add_row(m["name"], m["repo"], "[green]ready[/green]" if m["ready"] else "[red]missing[/red]")
    console.print(table)


@app.command()
def download(
    model: str = typer.Option(None, "--model", help="Only one: skillrouter-embedding | skillrouter-reranker (default: both)."),
    force: bool = typer.Option(False, "--force", help="Re-fetch even if already cached."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen; no network, no writes."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (stable contract)."),
) -> None:
    """Idempotently download the SkillRouter checkpoints to the HF cache."""
    resources = W.managed_resources()
    if model:
        selected = [r for r in resources if r["id"] == model]
        if not selected:
            console.print(f"[red]unknown model: {model}[/red]")
            raise typer.Exit(W.EXIT_USAGE)
        resources = selected

    results: list[dict] = []
    exit_code = W.EXIT_OK
    for r in resources:
        _from_cache, info, code = W.download_resource(r, force=force, dry_run=dry_run)
        results.append(info)
        if code not in (W.EXIT_OK, W.EXIT_USAGE) and exit_code == W.EXIT_OK:
            exit_code = code

    if json_out:
        typer.echo(json.dumps({"ok": exit_code == W.EXIT_OK, "results": results}, indent=2, ensure_ascii=False))
    else:
        for info in results:
            tag = "cached" if info.get("from_cache") else ("would download" if info.get("would_download") else "downloaded")
            console.print(f"[dim]{info.get('model')}[/dim] -> {info.get('dest', info.get('error', ''))} ({tag})")

    if exit_code != W.EXIT_OK:
        raise typer.Exit(exit_code)


@app.command()
def describe(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (stable contract)."),
) -> None:
    """Self-describe: capabilities, managed models, source path."""
    info = {
        "name": "skillrouter",
        "source": _source_path(),
        "capabilities": ["retrieve", "rerank", "route"],
        "models": [{"id": r["id"], "repo": r["repo"], "kind": "local-weights"} for r in W.managed_resources()],
    }
    if json_out:
        typer.echo(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        console.print(info)


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON (stable contract)."),
) -> None:
    """Report weight readiness. Missing weights are NOT fatal (exits 0 if the binary is healthy)."""
    payload = _status_payload()
    backend = W._module_available("transformers")
    if json_out:
        typer.echo(json.dumps({"binary_ok": True, "backend": {"transformers": backend}, **payload}, indent=2, ensure_ascii=False))
        return
    console.print(f"[dim]backend: transformers={'yes' if backend else 'no'}[/dim]")
    for m in payload["models"]:
        mark = "[green]✓[/green]" if m["ready"] else "[red]✗[/red]"
        console.print(f"  {mark} {m['name']}  {m['repo']}")


if __name__ == "__main__":
    app()
