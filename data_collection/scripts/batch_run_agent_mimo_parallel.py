#!/usr/bin/env python3
"""Parallel launcher for batch_run_agent_mimo.py.

Each task runs in its own subprocess — different processes execute different
tasks, never overlapping.

Usage:
    python scripts/batch_run_agent_mimo_parallel.py [--workers 4] [--dry-run]

Extra flags are forwarded to batch_run_agent_mimo.py:
    --count 30 --no-mock --supplement
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "batch_run_agent_mimo.py"

# Mirror AGENT_TASKS from batch_run_agent.py
AGENT_TASKS = (17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 40)

# Forwarded args (passed through to the batch script)
FORWARD_FLAGS = ("--count", "--no-mock", "--supplement")

# ---------------------------------------------------------------------------
# Subprocess bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class Job:
    task: int
    proc: subprocess.Popen
    logfile: Path
    start_ts: float

_jobs: Dict[int, Job] = {}
_proc_lock = __import__("threading").Lock()

def _handle_sigint(sig, frame):
    print("\nInterrupted. Terminating all child processes...", flush=True)
    with _proc_lock:
        for job in list(_jobs.values()):
            if job.proc.poll() is None:
                job.proc.terminate()
    # Give them a moment, then kill stragglers
    time.sleep(5)
    with _proc_lock:
        for job in list(_jobs.values()):
            if job.proc.poll() is None:
                job.proc.kill()
    sys.exit(130)

def _handle_sigterm(sig, frame):
    _handle_sigint(sig, frame)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def build_cmd(task: int, extra_args: List[str]) -> List[str]:
    """Build the command-line for a single task."""
    return [
        sys.executable, str(BATCH_SCRIPT),
        "--task", str(task),
        *extra_args,
    ]

def launch_job(task: int, extra_args: List[str], logs_dir: Path) -> Job:
    """Launch batch_run_agent_mimo.py for a single task, return a Job."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"task_{task:02d}.log"

    cmd = build_cmd(task, extra_args)
    print(f"  [task {task:02d}] launching: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=open(logfile, "w"),
        stderr=subprocess.STDOUT,
    )
    job = Job(task=task, proc=proc, logfile=logfile, start_ts=time.time())
    with _proc_lock:
        _jobs[task] = job
    return job

def reap_one() -> Optional[Job]:
    """Check all running jobs; return the first finished one (or None)."""
    with _proc_lock:
        for job in list(_jobs.values()):
            rc = job.proc.poll()
            if rc is not None:
                del _jobs[job.task]
                return job
    return None

def wait_for_slot(max_workers: int) -> Optional[Job]:
    """Block until at least one job finishes (or all finished).  Returns the
    finished job, or None when the queue is empty."""
    while True:
        with _proc_lock:
            if not _jobs:
                return None
        job = reap_one()
        if job is not None:
            return job
        # Nothing finished yet — wait briefly and retry (we can't
        # select/poll on multiple Popen objects cleanly in portable Python,
        # so a short sleep-loop is the pragmatic choice).
        time.sleep(2)

def run_parallel(tasks: List[int], extra_args: List[str],
                 max_workers: int, logs_dir: Path) -> Dict[int, int]:
    """Run *tasks* with at most *max_workers* in flight.  Return {task: rc}."""
    queue = list(tasks)
    running = 0
    results: Dict[int, int] = {}

    while queue or running > 0:
        # Fill slots
        while queue and running < max_workers:
            task = queue.pop(0)
            launch_job(task, extra_args, logs_dir)
            running += 1

        # Wait for a slot to free
        finished = wait_for_slot(max_workers)
        if finished is None:
            break  # all done

        rc = finished.proc.returncode
        elapsed = time.time() - finished.start_ts
        results[finished.task] = rc
        running -= 1

        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  [task {finished.task:02d}] finished: {status} ({elapsed:.0f}s)",
              flush=True)

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    parser = argparse.ArgumentParser(
        description="Launch batch_run_agent_mimo.py tasks in parallel")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent tasks (default: 4)")
    parser.add_argument("--dry-run", action="store_true")
    # Forward these to the batch script
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--no-mock", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()

    # Build forwarded args for the batch script
    extra_args = []
    if args.count:
        extra_args += ["--count", str(args.count)]
    if args.no_mock:
        extra_args.append("--no-mock")
    if args.supplement:
        extra_args.append("--supplement")

    tasks = sorted(AGENT_TASKS)
    logs_dir = PROJECT_ROOT / "episodes" / "Agent" / "cc_mimo" / "parallel_logs_30"

    print(f"=== Parallel AGENT batch (mimo) ===")
    print(f"  Tasks: {tasks}")
    print(f"  Workers: {args.workers}")
    print(f"  Extra args: {extra_args}")
    if args.dry_run:
        print(f"  [DRY RUN — will only print commands]")
    print()

    if args.dry_run:
        for t in tasks:
            cmd = build_cmd(t, extra_args)
            print(f"  {t:02d}: {' '.join(cmd)}")
        print(f"\n{len(tasks)} tasks, would use max {args.workers} workers.")
        return

    t0 = time.time()
    results = run_parallel(tasks, extra_args, args.workers, logs_dir)

    elapsed = time.time() - t0
    ok = sum(1 for rc in results.values() if rc == 0)
    fail = len(results) - ok

    print(f"\n{'='*60}")
    print(f"  PARALLEL BATCH DONE: {len(results)} tasks in {elapsed/60:.0f} min")
    print(f"  OK={ok} FAIL={fail}")
    print(f"{'='*60}\n")
    for t in sorted(results):
        status = "OK" if results[t] == 0 else f"FAIL({results[t]})"
        print(f"  task {t:02d}  {status}")
    print(f"\nLogs: {logs_dir}/")

if __name__ == "__main__":
    main()
