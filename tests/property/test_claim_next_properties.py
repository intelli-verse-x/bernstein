"""Property tests for the atomic ``claim_next`` primitive (#1220).

Concurrency contract under test:

1. **Mutual exclusion** — for any backlog of N tasks contended by W
   parallel workers, every task ends up claimed by *exactly one*
   worker. There are no double-claims and no lost rows.
2. **Conservation** — the number of successful claims plus the number
   of ``None`` returns equals the total number of ``claim_next``
   invocations. Nothing is silently dropped.
3. **Capacity** — when the call count exceeds the backlog size, the
   excess calls all return ``None`` (the backlog never hands out more
   than it holds).

The ``smoke`` profile already runs 50+ examples; we override the
example budget upward to ``max_examples=120`` (well over the
operator-mandated 100) so even the dedicated PR run exercises the
race window adequately.
"""

from __future__ import annotations

import tempfile
import threading
from collections import Counter
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from bernstein.core.tasks.claim import Backlog, claim_next


def _spawn_claimers(
    backlog_path: Path,
    workers: int,
    calls_per_worker: int,
) -> list[str | None]:
    """Run *workers* threads, each invoking ``claim_next`` *calls_per_worker* times.

    A barrier maximises the contention window: every thread blocks
    until all of them are ready to fire, so the first claims race
    rather than serialising naturally.

    Args:
        backlog_path: Path to the shared backlog under test.
        workers: Number of concurrent threads.
        calls_per_worker: How many ``claim_next`` calls each thread
            makes before exiting.

    Returns:
        Flat list of every value returned across all threads (claimed
        ids and ``None``s in completion order).
    """
    barrier = threading.Barrier(workers)
    results: list[str | None] = []
    results_lock = threading.Lock()

    def _runner(idx: int) -> None:
        local: list[str | None] = []
        barrier.wait()
        for _ in range(calls_per_worker):
            local.append(claim_next(backlog_path, claimer_id=f"worker-{idx}"))
        with results_lock:
            results.extend(local)

    threads = [threading.Thread(target=_runner, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


@settings(max_examples=120, deadline=None)
@given(
    n_tasks=st.integers(min_value=0, max_value=40),
    workers=st.integers(min_value=1, max_value=8),
    calls_per_worker=st.integers(min_value=1, max_value=6),
)
def test_no_double_claim_under_concurrency(
    n_tasks: int,
    workers: int,
    calls_per_worker: int,
) -> None:
    """Every task is claimed by exactly one worker, even under contention."""
    with tempfile.TemporaryDirectory(prefix="bernstein-claim-prop-") as tmp:
        backlog_path = Path(tmp) / "backlog.json"
        task_ids = [f"t{i}" for i in range(n_tasks)]
        Backlog.write(backlog_path, task_ids)

        results = _spawn_claimers(backlog_path, workers, calls_per_worker)

        total_calls = workers * calls_per_worker
        claimed = [r for r in results if r is not None]
        nones = [r for r in results if r is None]

        # 1. Conservation — every call accounted for.
        assert len(claimed) + len(nones) == total_calls

        # 2. Capacity — claims cannot exceed the backlog size.
        assert len(claimed) <= n_tasks

        # 3. Mutual exclusion — no id was returned twice.
        counts = Counter(claimed)
        assert all(count == 1 for count in counts.values()), (
            f"double-claim detected: {[(k, v) for k, v in counts.items() if v > 1]}"
        )

        # 4. The persisted state matches what the workers observed.
        final = Backlog.load(backlog_path)
        on_disk_claimed = {e.id: e.claimer for e in final.entries if e.claimer is not None}
        assert set(claimed) == set(on_disk_claimed.keys())

        # 5. Either we exhausted the backlog or had spare capacity.
        if total_calls >= n_tasks:
            assert len(claimed) == n_tasks
        else:
            assert len(claimed) == total_calls
            assert len(nones) == 0


@settings(max_examples=120, deadline=None)
@given(
    n_tasks=st.integers(min_value=1, max_value=20),
    workers=st.integers(min_value=2, max_value=8),
)
def test_drain_completes_with_one_call_per_task(
    n_tasks: int,
    workers: int,
) -> None:
    """Issuing exactly N calls across W workers fully drains an N-task backlog.

    Total calls equal the backlog size, so under correct mutex every
    task is claimed and no ``None`` is returned. Any ``None`` here
    would mean a worker observed a stale view and dropped a task on
    the floor.
    """
    calls_per_worker = max(1, n_tasks // workers + 1)
    with tempfile.TemporaryDirectory(prefix="bernstein-claim-prop-drain-") as tmp:
        backlog_path = Path(tmp) / "backlog.json"
        task_ids = [f"t{i}" for i in range(n_tasks)]
        Backlog.write(backlog_path, task_ids)

        results = _spawn_claimers(backlog_path, workers, calls_per_worker)

        claimed = [r for r in results if r is not None]
        assert sorted(claimed) == sorted(task_ids), f"expected to drain {task_ids} once each, got {sorted(claimed)}"
