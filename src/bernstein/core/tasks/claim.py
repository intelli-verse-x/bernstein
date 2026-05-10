"""Atomic claim primitive for race-safe shared backlogs (#1220).

Multi-worker crews that share one backlog need an atomic
``read-pick-mark`` operation so two workers never end up running the
same task. Without it the orchestrator either pre-assigns work (and
loses load-balancing) or silently double-spends compute when two
workers race the same row.

This module provides a lightweight, file-backed primitive for the
single-host case. It pairs an on-disk JSON document (the *backlog*)
with an OS-level advisory lock on a sibling ``.lock`` file. The lock
follows the same ``fcntl.flock`` / ``msvcrt.locking`` precedent already
used by :mod:`bernstein.core.persistence.file_locks` and the same
crash-safe write path as :mod:`bernstein.core.persistence.atomic_write`,
so concurrent readers either see the pre-claim or post-claim state —
never a torn document.

The primitive is single-host only. Multiple hosts pointing at the same
backlog over NFS/SMB cannot rely on advisory file locks (see the NFS
caveat in ``file_locks.py``); a Postgres-backed implementation behind
the same callable signature is left for a follow-up issue.

Typical usage::

    from pathlib import Path
    from bernstein.core.tasks.claim import Backlog, claim_next

    Backlog.write(Path("backlog.json"), ["a", "b", "c"])

    # In each worker:
    task_id = claim_next(Path("backlog.json"), claimer_id="worker-A")
    if task_id is None:
        return  # nothing left
    do_work(task_id)
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING

from bernstein.core.persistence.atomic_write import write_atomic_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-process advisory lock (mirrors persistence.file_locks)
# ---------------------------------------------------------------------------


if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _os_lock(fh: IO[bytes]) -> None:
        """Acquire an exclusive OS-level lock on *fh* (Windows)."""
        import time

        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                time.sleep(0.05)

    def _os_unlock(fh: IO[bytes]) -> None:
        """Release the OS-level lock on *fh* (Windows)."""
        from contextlib import suppress

        with suppress(OSError):
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _os_lock(fh: IO[bytes]) -> None:
        """Acquire an exclusive OS-level lock on *fh* (POSIX)."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _os_unlock(fh: IO[bytes]) -> None:
        """Release the OS-level lock on *fh* (POSIX)."""
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def _backlog_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a blocking exclusive OS lock at *lock_path*.

    The sentinel file is created on first use. Concurrent callers block
    until the lock is released or the holder process exits.

    Args:
        lock_path: Path to the ``.lock`` sentinel. Parents are created.

    Yields:
        ``None`` while the lock is held.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")  # noqa: SIM115 - released in finally below
    try:
        _os_lock(fh)
        try:
            yield
        finally:
            _os_unlock(fh)
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class BacklogEntry:
    """One row in the on-disk backlog.

    Attributes:
        id: Task identifier. Must be unique within a backlog.
        claimer: Worker that owns the row, or ``None`` while available.
    """

    id: str
    claimer: str | None = None


@dataclass
class Backlog:
    """In-memory view of a shared backlog file.

    Attributes:
        path: On-disk JSON document.
        entries: Ordered list of :class:`BacklogEntry`. The first
            unclaimed entry is the one ``claim_next`` will pick.
    """

    path: Path
    entries: list[BacklogEntry] = field(default_factory=list)

    @property
    def lock_path(self) -> Path:
        """Path to the sibling ``.lock`` sentinel for this backlog."""
        return self.path.with_suffix(self.path.suffix + ".lock")

    @classmethod
    def load(cls, path: Path) -> Backlog:
        """Load a backlog from *path*.

        A missing file is treated as an empty backlog so the caller
        does not have to special-case bootstrap.

        Args:
            path: On-disk JSON document.

        Returns:
            Hydrated :class:`Backlog`.

        Raises:
            ValueError: If the on-disk payload is not a JSON list of
                ``{id, claimer}`` rows.
        """
        if not path.exists():
            return cls(path=path, entries=[])
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            msg = f"backlog at {path} must be a JSON array, got {type(raw).__name__}"
            raise ValueError(msg)
        entries = [BacklogEntry(id=str(row["id"]), claimer=row.get("claimer")) for row in raw]
        return cls(path=path, entries=entries)

    def save(self) -> None:
        """Persist the current state via the atomic-write helper."""
        write_atomic_json(
            self.path,
            [{"id": e.id, "claimer": e.claimer} for e in self.entries],
        )

    @classmethod
    def write(cls, path: Path, task_ids: Iterable[str]) -> Backlog:
        """Create or overwrite a backlog at *path* with *task_ids*.

        Convenience for setup. Existing claimer state is dropped.

        Args:
            path: Destination JSON document.
            task_ids: Ordered iterable of unique task identifiers.

        Returns:
            The freshly-written :class:`Backlog`.
        """
        backlog = cls(path=path, entries=[BacklogEntry(id=str(tid)) for tid in task_ids])
        backlog.save()
        return backlog


# ---------------------------------------------------------------------------
# Atomic claim primitive
# ---------------------------------------------------------------------------


def claim_next(backlog_path: Path, claimer_id: str) -> str | None:
    """Atomically claim the next available task in *backlog_path*.

    Steps, all under a cross-process exclusive lock:

    1. Open or create the sibling ``.lock`` sentinel and ``flock`` it.
    2. Read the backlog (treating a missing file as empty).
    3. Pick the first entry whose ``claimer`` is ``None``.
    4. Stamp ``claimer_id`` on it and persist via ``write_atomic_json``.
    5. Return the claimed task id.

    If the backlog is empty, missing, or all entries are already
    claimed, the function returns ``None`` without mutating disk
    state.

    Args:
        backlog_path: Path to the backlog JSON document.
        claimer_id: Stable identifier of the requesting worker. Stored
            on the row so failure-recovery tools can attribute work.

    Returns:
        The claimed task id, or ``None`` if nothing is available.
    """
    backlog = Backlog(path=backlog_path)
    with _backlog_lock(backlog.lock_path):
        backlog = Backlog.load(backlog_path)
        for entry in backlog.entries:
            if entry.claimer is None:
                entry.claimer = claimer_id
                backlog.save()
                logger.debug(
                    "claim_next: %s -> %s (backlog=%s)",
                    entry.id,
                    claimer_id,
                    backlog_path,
                )
                return entry.id
    return None


__all__ = [
    "Backlog",
    "BacklogEntry",
    "claim_next",
]
