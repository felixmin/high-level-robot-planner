import multiprocessing as mp
import time
from pathlib import Path

from common.adapters.openx_local_indexed_full import _index_build_lock


def _lock_worker(index_root: str, hold_sec: float, queue: mp.Queue) -> None:
    start = time.monotonic()
    with _index_build_lock(Path(index_root)):
        acquired = time.monotonic()
        queue.put(("acquired", acquired - start))
        if hold_sec > 0:
            time.sleep(hold_sec)


def test_index_build_lock_serializes_processes(tmp_path: Path) -> None:
    index_root = tmp_path / "index_key"
    queue: mp.Queue = mp.Queue()

    holder = mp.Process(target=_lock_worker, args=(str(index_root), 1.0, queue))
    waiter = mp.Process(target=_lock_worker, args=(str(index_root), 0.0, queue))

    holder.start()
    time.sleep(0.1)
    waiter.start()

    holder.join(timeout=10)
    waiter.join(timeout=10)

    assert holder.exitcode == 0
    assert waiter.exitcode == 0

    acquired_delays = [queue.get(timeout=2)[1] for _ in range(2)]
    # One process acquires immediately; the other waits for lock release.
    assert min(acquired_delays) < 0.5
    assert max(acquired_delays) >= 0.8

