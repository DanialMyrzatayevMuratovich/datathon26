"""POSIX worker guardian: bound lifetime even if the API or native inference hangs."""

import os
import signal
import sys
import time


def watch(parent_pid: int, worker_pid: int, deadline: float) -> None:
    # The manager creates a new session whose group ID equals the worker PID.
    # Refuse to signal anything outside the process group we ourselves belong to.
    if os.name == "nt" or worker_pid <= 1 or os.getpgrp() != worker_pid:
        raise RuntimeError("Watchdog requires the worker's isolated POSIX process group")
    while time.monotonic() < deadline:
        if os.getppid() != worker_pid:
            # Worker finished. Its remaining child processes must also be stopped.
            break
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    os.killpg(worker_pid, signal.SIGKILL)


if __name__ == "__main__":
    watch(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
