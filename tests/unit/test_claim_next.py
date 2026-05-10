"""Unit tests for the file-backed ``claim_next`` primitive (#1220).

Covers the obvious sequential cases:

* Empty / missing backlog returns ``None``.
* Backlog with all entries claimed returns ``None``.
* Successive ``claim_next`` calls hand out distinct entries in
  insertion order.
* Each claim stamps the requested ``claimer_id`` on disk.

Concurrency invariants live in the matching property test
(``tests/property/test_claim_next_properties.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from bernstein.core.tasks.claim import Backlog, claim_next


def test_claim_next_missing_backlog_returns_none(tmp_path: Path) -> None:
    """A missing backlog file is treated as empty."""
    backlog_path = tmp_path / "nope.json"
    assert claim_next(backlog_path, claimer_id="w1") is None


def test_claim_next_empty_backlog_returns_none(tmp_path: Path) -> None:
    """An explicitly-empty backlog returns None without mutating disk."""
    backlog_path = tmp_path / "backlog.json"
    Backlog.write(backlog_path, [])
    before = backlog_path.read_bytes()

    assert claim_next(backlog_path, claimer_id="w1") is None

    assert backlog_path.read_bytes() == before


def test_claim_next_all_claimed_returns_none(tmp_path: Path) -> None:
    """When every row has a claimer, claim_next returns None."""
    backlog_path = tmp_path / "backlog.json"
    Backlog.write(backlog_path, ["t1", "t2"])

    assert claim_next(backlog_path, claimer_id="w1") == "t1"
    assert claim_next(backlog_path, claimer_id="w2") == "t2"
    assert claim_next(backlog_path, claimer_id="w3") is None


def test_claim_next_marks_claimer_on_disk(tmp_path: Path) -> None:
    """The claimed row carries the requested claimer_id after the call."""
    backlog_path = tmp_path / "backlog.json"
    Backlog.write(backlog_path, ["t1", "t2"])

    claimed = claim_next(backlog_path, claimer_id="worker-A")
    assert claimed == "t1"

    rows = json.loads(backlog_path.read_text(encoding="utf-8"))
    assert rows == [
        {"id": "t1", "claimer": "worker-A"},
        {"id": "t2", "claimer": None},
    ]


def test_claim_next_skips_already_claimed_entries(tmp_path: Path) -> None:
    """Pre-claimed entries are skipped so workers never double-claim."""
    backlog_path = tmp_path / "backlog.json"
    backlog_path.write_text(
        json.dumps(
            [
                {"id": "t1", "claimer": "ghost"},
                {"id": "t2", "claimer": None},
            ],
        ),
        encoding="utf-8",
    )

    assert claim_next(backlog_path, claimer_id="worker-B") == "t2"


def test_backlog_write_roundtrip(tmp_path: Path) -> None:
    """Backlog.write + Backlog.load round-trips without claimer state."""
    backlog_path = tmp_path / "backlog.json"
    Backlog.write(backlog_path, ["a", "b", "c"])

    loaded = Backlog.load(backlog_path)
    assert [e.id for e in loaded.entries] == ["a", "b", "c"]
    assert all(e.claimer is None for e in loaded.entries)
