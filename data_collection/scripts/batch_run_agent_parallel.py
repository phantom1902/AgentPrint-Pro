#!/usr/bin/env python3
"""Parallel batch runner for agent tasks — multiple episodes run concurrently.

Each episode has its own network namespace (netns) + veth pair, so network
traffic is fully isolated between concurrent episodes. Filesystem isolation
is handled by sandbox_bwrap.sh (each episode has its own bind-mounted workspace).

Usage:
    python scripts/batch_run_agent_parallel.py --workers 4

Requires mock services to be started externally:
    bash scripts/env/mock_services.sh start
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

def _available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return 4.0  # guess

def _safe_worker_limit():
    """Return a memory-aware default worker count.

    Each concurrent episode runs a Claude Code (Node.js) process that can
    consume 500MB–2GB during active tool use.  We reserve ~4GB per worker
    to be safe, so a 7.1Gi machine defaults to 1–2 workers regardless of
    CPU count.
    """
    avail_gb = _available_gb()
    return max(1, min(cpu_count(), int(avail_gb // 4)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
EPISODES_DIR = PROJECT_ROOT / "episodes"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNNER_DIR = SCRIPTS_DIR / "runner"
ENV_DIR = SCRIPTS_DIR / "env"

SAMPLES_PER_TASK = 10
SKIP_TASKS = {19, 22}
# AGENT_TASKS = {15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}
# AGENT_TASKS = {18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}
AGENT_TASKS = {15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 40}
CLI_TEXT_TASKS = {34, 35, 36, 37, 39, 40}
DEFAULT_HARNESS = "claude_code_deepseek"
DEFAULT_TIMEOUT = 600

# ── helpers (reused from batch_run_agent.py) ──────────────────────────

def find_task_dirs(task_set):
    tasks = {}
    for d in sorted(TASKS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        parts = d.name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            num = int(parts[0])
        except ValueError:
            continue
        if num in SKIP_TASKS or num not in task_set:
            continue
        tasks[num] = d
    return tasks

def find_instance_files(task_dir, limit=SAMPLES_PER_TASK):
    inst_dir = task_dir / "instances"
    if not inst_dir.exists():
        return []
    return sorted(inst_dir.glob("sample_*.yaml"))[:limit]

def episode_already_done(episode_id):
    resp = EPISODES_DIR / episode_id / "cli_response.json"
    if not resp.exists():
        return False
    try:
        d = json.loads(resp.read_text())
        return d.get("completed", False)
    except Exception:
        return False

def clean_netns(episode_id):
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
    )

def prune_episode(episode_id, rc):
    ep_dir = EPISODES_DIR / episode_id
    really_ok = (rc == 0 and (ep_dir / "cli_response.json").exists())
    if not really_ok:
        return
    for fname in ("orchestrator_stdout.log", "orchestrator_stderr.log",
                  "stderr.log", "stdout.log"):
        (ep_dir / fname).unlink(missing_ok=True)

# ── parallel worker ───────────────────────────────────────────────────

def run_one_episode(args_dict):
    """Worker entry point for multiprocessing Pool. Runs a single episode."""
    episode_id = args_dict["episode_id"]
    task_dir = args_dict["task_dir"]
    instance_path = args_dict["instance_path"]
    harness = args_dict["harness"]
    timeout_s = args_dict["timeout_s"]
    dry_run = args_dict.get("dry_run", False)
    no_mock = args_dict.get("no_mock", True)

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUNNER_DIR / "run_episode.py"), episode_id,
        "--runner", "cli",
        "--instance", str(instance_path),
        "--harness", harness,
        "--timeout", str(timeout_s),
    ]
    if no_mock:
        cmd.append("--no-mock")

    print(f"  [{episode_id}] START {task_dir.name}/{instance_path.name}  harness={harness}", flush=True)

    if dry_run:
        return {"episode_id": episode_id, "rc": 0, "elapsed": 0}

    subprocess.run(["sudo", "chown", "-R",
                    f"{os.getuid()}:{os.getgid()}", str(ep_dir)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)

    t0 = time.time()
    rc = -1
    try:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT,
            stdout=open(ep_dir / "orchestrator_stdout.log", "w"),
            stderr=open(ep_dir / "orchestrator_stderr.log", "w"),
        )
        try:
            rc = proc.wait(timeout=timeout_s + 120)
        except subprocess.TimeoutExpired:
            kill_netns(episode_id)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            rc = -1
    finally:
        pass

    elapsed = time.time() - t0
    status = "OK" if rc == 0 else "TIMEOUT" if rc == -1 else f"FAIL({rc})"
    print(f"  [{episode_id}] DONE {status} ({elapsed:.0f}s)", flush=True)

    # Post-processing
    if not dry_run:
        clean_netns(episode_id)
        prune_episode(episode_id, rc)

    # Write batch meta
    meta = {
        "episode_id": episode_id, "task": task_dir.name,
        "instance": str(instance_path), "exit_code": rc,
        "elapsed_s": elapsed, "timeout_s": timeout_s,
        "harness": harness,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"episode_id": episode_id, "rc": rc, "elapsed": elapsed}

def kill_netns(episode_id):
    ns = f"fb_{episode_id}"
    host_if = f"veth_{episode_id}_host"
    subprocess.run(
        f"sudo ip netns pids {ns} 2>/dev/null | xargs -r sudo kill -9 2>/dev/null; "
        f"sudo ip netns del {ns} 2>/dev/null; "
        f"sudo ip link del {host_if} 2>/dev/null; "
        f"sudo rm -rf /etc/netns/{ns} 2>/dev/null",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
    )

# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel batch run all AGENT episodes")
    safe_default = _safe_worker_limit()
    parser.add_argument("--workers", "-w", type=int, default=safe_default,
                        help=f"Number of parallel workers (default: {safe_default}, CPU count: {cpu_count()})")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--harness", default=DEFAULT_HARNESS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Seconds per episode (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--count", type=int, default=None,
                        help=f"Max samples per task (default: {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--supplement", action="store_true",
                        help="Skip already-successful episodes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Safety: warn if user forces more workers than memory can support
    if args.workers > safe_default:
        print(f"WARNING: {args.workers} workers requested but system has only "
              f"~{_available_gb():.1f}GiB available memory "
              f"(recommended: {safe_default}). Expect thrashing or OOM.",
              flush=True)

    print(f"=== Parallel Agent Batch: {args.workers} workers ===", flush=True)
    print(f"Harness: {args.harness} | Timeout: {args.timeout}s | Supplement: {args.supplement}", flush=True)

    # ── Build work queue ──
    tasks = find_task_dirs(AGENT_TASKS | CLI_TEXT_TASKS)
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}

    all_jobs = []
    for task_num in sorted(tasks):
        task_dir = tasks[task_num]
        if args.count is None:
            instances = find_instance_files(task_dir)   # uses SAMPLES_PER_TASK
        elif args.count == 0:
            instances = find_instance_files(task_dir, limit=None)  # all
        else:
            instances = find_instance_files(task_dir, limit=args.count)
        for idx, inst in enumerate(instances):
            eid = f"A{task_num:02d}{idx:02d}"
            if args.supplement and episode_already_done(eid):
                print(f"  [{eid}] SKIP (already done)", flush=True)
                continue
            all_jobs.append({
                "episode_id": eid,
                "task_dir": task_dir,
                "instance_path": inst,
                "harness": args.harness,
                "timeout_s": args.timeout,
                "dry_run": args.dry_run,
                "no_mock": True,   # parallel mode: mock must be external
            })

    if not all_jobs:
        print("No episodes to run.", flush=True)
        return

    total = len(all_jobs)
    print(f"\n{total} episodes to run across {args.workers} workers\n", flush=True)

    if args.dry_run:
        for j in all_jobs:
            print(f"  [{j['episode_id']}] {j['task_dir'].name}/{j['instance_path'].name}", flush=True)
        return

    # ── Cleanup leftover state before start ──
    print("Cleaning up leftover network state...", flush=True)
    subprocess.run(
        "sudo ip netns list 2>/dev/null | grep 'fb_A' | while read ns; do "
        "sudo ip netns del \"$ns\" 2>/dev/null; done; "
        "ip link show 2>/dev/null | grep -oP 'veth_A[^ @]+' | while read v; do "
        "sudo ip link del \"$v\" 2>/dev/null; done; "
        "sudo rm -rf /etc/netns/fb_A* 2>/dev/null",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
    )
    print("  Done.\n", flush=True)

    # ── Run in parallel ──
    n_workers = min(args.workers, total)
    t0 = time.time()
    ok = fail = timeout_n = 0

    def _worker_init():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    pool = Pool(processes=n_workers, initializer=_worker_init, maxtasksperchild=1)

    # Submit all jobs asynchronously
    async_results = [pool.apply_async(run_one_episode, (job,)) for job in all_jobs]
    pool.close()

    try:
        # Wait for all with periodic progress
        while async_results:
            still_pending = []
            for ar in async_results:
                if ar.ready():
                    try:
                        result = ar.get(timeout=0.1)
                    except Exception:
                        result = {"episode_id": "?", "rc": -1, "elapsed": 0}
                    eid = result.get("episode_id", "?")
                    rc = result.get("rc", -1)
                    if rc == 0:
                        ok += 1
                    elif rc == -1:
                        timeout_n += 1
                    else:
                        fail += 1
                else:
                    still_pending.append(ar)
            async_results = still_pending
            done = ok + fail + timeout_n
            active = len(still_pending) - sum(1 for ar in still_pending if ar.ready())
            actual_active = max(n_workers, len(still_pending))
            eta = (time.time() - t0) / max(done, 1) * (total - done)
            print(f"  [{done}/{total}] OK={ok} FAIL={fail} TIMEOUT={timeout_n}  "
                  f"active~{min(n_workers, len(still_pending))}  ETA={eta/60:.0f}m", flush=True)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nInterrupted. Killing all workers...", flush=True)
        pool.terminate()
        pool.join()
        # Kill any remaining run_episode.py processes
        subprocess.run(
            "pgrep -f 'run_episode.py' 2>/dev/null | xargs -r kill -9 2>/dev/null",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        print("All killed.", flush=True)
        sys.exit(130)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PARALLEL BATCH DONE: {total} episodes in {elapsed/60:.0f} min "
          f"({args.workers} workers)")
    print(f"  OK={ok} FAIL={fail} TIMEOUT={timeout_n}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
