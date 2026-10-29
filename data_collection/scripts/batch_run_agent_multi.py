#!/usr/bin/env python3
"""Task-level parallel launcher for batch_run_agent.py.

This follows the same parallelization model as batch_run_agent_mimo_parallel.py:
the parent process launches one batch_run_agent.py subprocess per task, and each
task subprocess runs its samples serially. The original per-task batch output is
kept intact in a dedicated task log.

Examples:
    python scripts/batch_run_agent_multi.py --tasks 18,20-23 --workers 3
    python scripts/batch_run_agent_multi.py --tasks 18 --count 10 --supplement
    python scripts/batch_run_agent_multi.py --all --workers 4 --no-mock
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
BATCH_SCRIPT = PROJECT_ROOT / "scripts" / "batch_run_agent.py"
MOCK_SERVICES = PROJECT_ROOT / "scripts" / "env" / "mock_services.sh"
EPISODES_DIR = PROJECT_ROOT / "episodes" / "Agent" / "cc_dpsk"

SAMPLES_PER_TASK = 10
DEFAULT_HARNESS = "claude_code_deepseek"
DEFAULT_TIMEOUT = 3600
SKIP_TASKS = {19, 22}
AGENT_TASKS = {
    15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 37, 40,
}


@dataclass
class Job:
    task: int
    proc: subprocess.Popen
    logfile: Path
    log_handle: TextIO
    start_ts: float


_jobs: Dict[int, Job] = {}
_proc_lock = threading.Lock()
_mock_started = False


def available_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return 4.0
    return 4.0


def safe_worker_limit() -> int:
    return max(1, min(os.cpu_count() or 1, int(available_gb() // 4)))


def parse_task_spec(spec: str) -> set[int]:
    tasks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid task range: {part}")
            tasks.update(range(start, end + 1))
        else:
            tasks.add(int(part))
    return tasks


def find_task_dirs(task_set: set[int]) -> dict[int, Path]:
    tasks: dict[int, Path] = {}
    for path in sorted(TASKS_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            num = int(path.name.split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        if num in SKIP_TASKS or num not in task_set:
            continue
        tasks[num] = path
    return tasks


def cleanup_leftovers() -> None:
    subprocess.run(
        "sudo ip netns list 2>/dev/null | grep 'fb_A' | while read ns; do "
        "sudo ip netns del \"$ns\" 2>/dev/null; done; "
        "ip link show 2>/dev/null | grep -oP 'veth_A[^ @]+' | while read v; do "
        "sudo ip link del \"$v\" 2>/dev/null; done; "
        "sudo rm -rf /etc/netns/fb_A* 2>/dev/null",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )


def start_mock_services(external_mock: bool, dry_run: bool) -> bool:
    if dry_run or external_mock:
        return False
    print("Starting shared mock services...", flush=True)
    result = subprocess.run(["bash", str(MOCK_SERVICES), "start"], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError("Failed to start mock services")
    return True


def stop_mock_services() -> None:
    global _mock_started
    if _mock_started:
        print("Stopping shared mock services...", flush=True)
        subprocess.run(["bash", str(MOCK_SERVICES), "stop"], cwd=PROJECT_ROOT)
        _mock_started = False


def build_cmd(task: int, args: argparse.Namespace, include_dry_run: bool = False) -> List[str]:
    cmd = [
        sys.executable,
        str(BATCH_SCRIPT),
        "--task",
        str(task),
        "--harness",
        args.harness,
        "--timeout",
        str(args.timeout),
        "--count",
        str(args.count),
        "--no-mock",
        "--skip-initial-cleanup",
    ]
    if args.supplement:
        cmd.append("--supplement")
    if args.keep_logs:
        cmd.append("--keep-logs")
    if include_dry_run:
        cmd.append("--dry-run")
    return cmd


def launch_job(task: int, args: argparse.Namespace, logs_dir: Path) -> Job:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"task_{task:02d}.log"
    cmd = build_cmd(task, args)
    print(f"  [task {task:02d}] launching: {' '.join(cmd)}", flush=True)

    log_handle = open(logfile, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = Job(task=task, proc=proc, logfile=logfile, log_handle=log_handle, start_ts=time.time())
    with _proc_lock:
        _jobs[task] = job
    return job


def reap_one() -> Optional[Job]:
    with _proc_lock:
        for job in list(_jobs.values()):
            rc = job.proc.poll()
            if rc is not None:
                del _jobs[job.task]
                job.log_handle.close()
                return job
    return None


def wait_for_slot() -> Optional[Job]:
    while True:
        with _proc_lock:
            if not _jobs:
                return None
        job = reap_one()
        if job is not None:
            return job
        time.sleep(2)


def terminate_children() -> None:
    with _proc_lock:
        jobs = list(_jobs.values())
    for job in jobs:
        if job.proc.poll() is None:
            job.proc.terminate()
    time.sleep(5)
    for job in jobs:
        if job.proc.poll() is None:
            job.proc.kill()
        job.log_handle.close()


def handle_signal(signum, frame) -> None:
    print("\nInterrupted. Terminating task subprocesses...", flush=True)
    terminate_children()
    stop_mock_services()
    cleanup_leftovers()
    sys.exit(130)


def run_parallel(tasks: List[int], args: argparse.Namespace, logs_dir: Path) -> Dict[int, dict]:
    queue = list(tasks)
    running = 0
    results: Dict[int, dict] = {}

    while queue or running > 0:
        while queue and running < args.workers:
            launch_job(queue.pop(0), args, logs_dir)
            running += 1

        finished = wait_for_slot()
        if finished is None:
            break

        rc = finished.proc.returncode
        elapsed = time.time() - finished.start_ts
        status = "OK" if rc == 0 else f"FAIL({rc})"
        results[finished.task] = {
            "exit_code": rc,
            "elapsed_s": elapsed,
            "log": str(finished.logfile),
        }
        running -= 1
        print(f"  [task {finished.task:02d}] finished: {status} ({elapsed:.0f}s) "
              f"log={finished.logfile}", flush=True)

    return results


def main() -> int:
    global _mock_started

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    safe_default = safe_worker_limit()
    parser = argparse.ArgumentParser(
        description="Run multiple batch_run_agent.py tasks in parallel")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--tasks", help="Task list/ranges, e.g. 18,20-23")
    task_group.add_argument("--all", action="store_true", help="Run all supported agent tasks")
    parser.add_argument("--workers", "-j", type=int, default=safe_default,
                        help=f"Concurrent tasks (default: {safe_default}, memory-aware)")
    parser.add_argument("--count", type=int, default=SAMPLES_PER_TASK,
                        help=f"Samples per task (default: {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--harness", default=DEFAULT_HARNESS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--external-mock", "--no-mock", dest="external_mock",
                        action="store_true",
                        help="Do not manage mock services; assume they are already running")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-logs", action="store_true",
                        help="Forward to batch_run_agent.py; keep stdout/stderr logs on success")
    parser.add_argument("--logs-dir", type=Path, default=None,
                        help="Directory for per-task logs")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.all or not args.tasks:
        requested_tasks = set(AGENT_TASKS)
    else:
        requested_tasks = parse_task_spec(args.tasks)

    supported_requested = requested_tasks & AGENT_TASKS
    unsupported = sorted(requested_tasks - AGENT_TASKS - SKIP_TASKS)
    if unsupported:
        print(f"WARNING: unsupported agent tasks ignored: {unsupported}", flush=True)

    task_dirs = find_task_dirs(supported_requested)
    missing = sorted(supported_requested - set(task_dirs))
    if missing:
        print(f"WARNING: no task directories found for: {missing}", flush=True)

    tasks = sorted(task_dirs)
    if not tasks:
        print("No tasks to run.", flush=True)
        return 0

    if args.workers > safe_default:
        print(f"WARNING: requested {args.workers} workers, but available memory is "
              f"~{available_gb():.1f}GiB; recommended <= {safe_default}.", flush=True)

    logs_dir = args.logs_dir or (EPISODES_DIR / f"new_new_parallel_logs_{args.count if args.count else 'all'}")

    print("=== Multi-task agent batch ===", flush=True)
    print(f"Tasks: {tasks}", flush=True)
    print(f"Workers: {args.workers} | Harness: {args.harness} | Timeout: {args.timeout}s", flush=True)
    print(f"Task logs: {logs_dir}", flush=True)
    print("Parallelism: task-level subprocesses; samples within each task stay serial.", flush=True)
    if args.external_mock:
        print("Mocks: external/no-mock mode; parent will not start or stop mocks.", flush=True)
    if args.dry_run:
        print("[DRY RUN MODE]", flush=True)
        for task in tasks:
            print(f"  {task:02d}: {' '.join(build_cmd(task, args, include_dry_run=True))}", flush=True)
        return 0

    cleanup_leftovers()
    _mock_started = start_mock_services(args.external_mock, args.dry_run)

    t0 = time.time()
    try:
        results = run_parallel(tasks, args, logs_dir)
    finally:
        stop_mock_services()

    elapsed = time.time() - t0
    ok = sum(1 for item in results.values() if item["exit_code"] == 0)
    fail = len(results) - ok
    report = {
        "mode": "agent_task_parallel",
        "tasks": tasks,
        "workers": args.workers,
        "count": args.count,
        "harness": args.harness,
        "timeout_s": args.timeout,
        "elapsed_s": elapsed,
        "ok": ok,
        "fail": fail,
        "logs_dir": str(logs_dir),
        "results": results,
    }

    report_path = EPISODES_DIR / "agent_multi_parallel_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  MULTI-TASK BATCH DONE: {len(results)} tasks in {elapsed / 60:.1f} min")
    print(f"  OK={ok} FAIL={fail}")
    print(f"  Logs: {logs_dir}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 60}")
    for task in tasks:
        rc = results.get(task, {}).get("exit_code")
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  task {task:02d}  {status}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
