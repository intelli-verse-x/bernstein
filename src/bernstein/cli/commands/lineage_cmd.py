"""``bernstein lineage <file>:<line>`` -- walk the artifact lineage chain.

Reads :class:`bernstein.core.persistence.lineage.LineageRecord` entries
from the run's WAL and prints the producing agent, rendered-prompt SHA,
input artifacts, and cost for the requested file/line. Single-run only;
cross-run stitching is out of scope for v1.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.table import Table

from bernstein.cli.helpers import console


def _parse_target(target: str) -> tuple[str, int | None]:
    """Split ``"path/to/file.py:42"`` into ``("path/to/file.py", 42)``.

    A bare path returns ``(path, None)`` -- the lookup then matches
    every record for the file regardless of line.
    """
    if ":" not in target:
        return target, None
    path, _, suffix = target.rpartition(":")
    if not path:
        return target, None
    try:
        line = int(suffix)
    except ValueError:
        return target, None
    return path, line


@click.command("lineage")
@click.argument("target", required=True)
@click.option(
    "--workdir",
    "-w",
    type=click.Path(file_okay=False, exists=True),
    default=".",
    show_default=True,
    help="Project root containing .sdd/.",
)
@click.option(
    "--run",
    "run_id",
    default=None,
    help="Restrict to a single run id (default: all runs in the WAL directory).",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of records to display.",
)
def lineage_cmd(target: str, workdir: str, run_id: str | None, limit: int) -> None:
    """Walk the lineage chain backwards from ``<file>[:<line>]``.

    \b
    Examples:
      bernstein lineage src/foo.py:42
      bernstein lineage src/foo.py
      bernstein lineage src/foo.py:42 --run r-2026-05-05
    """
    from bernstein.core.persistence.lineage import LineageReader

    sdd_dir = Path(workdir).resolve() / ".sdd"
    if not sdd_dir.is_dir():
        console.print(f"[red]No .sdd directory at[/red] {sdd_dir}")
        raise SystemExit(1)

    path, line = _parse_target(target)

    reader = LineageReader(sdd_dir)
    records = reader.lookup(path, line, run_id=run_id)

    if not records:
        console.print(f"[yellow]No lineage records for[/yellow] {target}")
        return

    records = records[-limit:]

    where = f"{path}:{line}" if line is not None else path
    console.print()
    console.print(f"[bold]Lineage trail for[/bold] {where} ({len(records)} record(s))")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim", no_wrap=True)
    table.add_column("Producer", no_wrap=True)
    table.add_column("Run", no_wrap=True)
    table.add_column("Prompt SHA", style="dim", no_wrap=True)
    table.add_column("Inputs", overflow="fold")
    table.add_column("Model", no_wrap=True)
    table.add_column("Tokens", justify="right")
    table.add_column("Cost USD", justify="right")

    for record in records:
        ts = f"{record.timestamp:.0f}" if record.timestamp else "—"
        inputs_str = ", ".join(a.path for a in record.inputs) or "—"
        prompt_short = record.prompt_sha[:12] + "…" if record.prompt_sha else "—"
        table.add_row(
            ts,
            record.producer.agent_id,
            record.producer.run_id,
            prompt_short,
            inputs_str,
            record.model or "—",
            str(record.tokens),
            f"{record.cost_usd:.4f}",
        )

    console.print(table)
    console.print()
