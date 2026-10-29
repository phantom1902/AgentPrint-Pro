#!/usr/bin/env python3
"""Batch run FlowBenchmark agent tasks with Qwen Code.

Each sample = independent episode with isolated netns + workspace sandbox.
Serial execution (one episode at a time), matching batch_run_agent.py pattern.

Usage:
    # All tasks, 10 samples each
    python scripts/batch_run_qwen.py

    # Specific task
    python scripts/batch_run_qwen.py --task 17

    # Skip already-successful episodes (supplement mode)
    python scripts/batch_run_qwen.py --supplement

    # Run all samples for each task (no limit)
    python scripts/batch_run_qwen.py --count 0
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
EPISODES_DIR = PROJECT_ROOT / "episodes" / "Agent" / "qwen"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ENV_DIR = SCRIPTS_DIR / "env"

RUN_QWEN_SCRIPT = SCRIPTS_DIR / "run_qwen_episode.py"

SAMPLES_PER_TASK = 10
SKIP_TASKS = {19, 22}

# Same agent-only tasks as batch_run_agent.py
AGENT_TASKS = {15,16,17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 40}

DEFAULT_MODEL = "qwen3.6-plus"
DEFAULT_TIMEOUT = 1800  # 30 min per episode

_child_proc = None

def _handle_sigint(sig, frame):
    if _child_proc and _child_proc.poll() is None:
        _child_proc.terminate()
        try:
            _child_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _child_proc.kill()
    print("\nInterrupted. Cleaning up...", flush=True)
    sys.exit(130)

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

def clean_netns(episode_id):
    """Clean up a netns (best-effort)."""
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=30,
    )

def run_one_qwen_episode(episode_id, task_dir, instance_path,
                         model, timeout_s, dry_run=False,
                         no_mock=False, no_docker=False):
    cmd = [
        sys.executable, str(RUN_QWEN_SCRIPT), episode_id,
        "--instance", str(instance_path),
        "--model", model,
        "--timeout", str(timeout_s),
    ]
    if no_mock:
        cmd.append("--no-mock")
    if no_docker:
        cmd.append("--no-docker")

    print(f"  [{episode_id}] {task_dir.name}/{instance_path.name} "
          f"model={model}", flush=True)

    if dry_run:
        return 0

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(ep_dir)],
        capture_output=True, timeout=5,
    )

    t0 = time.time()
    try:
        global _child_proc
        _child_proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT,
            stdout=open(ep_dir / "orchestrator_stdout.log", "w"),
            stderr=open(ep_dir / "orchestrator_stderr.log", "w"),
        )
        try:
            rc = _child_proc.wait(timeout=timeout_s + 120)
        except subprocess.TimeoutExpired:
            # Kill netns aggressively
            _child_proc.terminate()
            try:
                _child_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _child_proc.kill()
            clean_netns(episode_id)
            rc = -1
            status = "TIMEOUT"
        else:
            status = "OK" if rc == 0 else f"FAIL({rc})"
    finally:
        _child_proc = None

    elapsed = time.time() - t0

    meta = {
        "episode_id": episode_id, "task": task_dir.name,
        "instance": str(instance_path), "exit_code": rc,
        "elapsed_s": elapsed, "timeout_s": timeout_s,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    -> {status} ({elapsed:.0f}s)", flush=True)
    return rc

def prune_episode(episode_id, rc):
    """On success, remove debug logs to save space."""
    ep_dir = EPISODES_DIR / episode_id
    really_ok = (rc == 0 and (ep_dir / "cli_response.json").exists())
    if not really_ok:
        return
    for fname in ("orchestrator_stdout.log", "orchestrator_stderr.log",
                  "stderr.log", "stdout.log"):
        (ep_dir / fname).unlink(missing_ok=True)

def episode_already_done(episode_id):
    """Check if a Qwen episode already completed successfully."""
    resp = EPISODES_DIR / episode_id / "cli_response.json"
    if not resp.exists():
        return False
    try:
        d = json.loads(resp.read_text())
        return d.get("completed", False)
    except Exception:
        return False

def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Batch run Qwen episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", type=int, default=0,
                        help="Run only one task (e.g. --task 17)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Qwen model (default {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Seconds per episode (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--count", type=int, default=None,
                        help=f"Max samples per task (default: {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--no-mock", action="store_true",
                        help="Skip starting mock services (use if already running)")
    parser.add_argument("--no-docker", action="store_true", default=False,
                        help="Skip Docker lifecycle for task 40 (Juice Shop)")
    parser.add_argument("--supplement", action="store_true",
                        help="Skip already-successful episodes, only re-run failed/missing")
    args = parser.parse_args()

    tasks = find_task_dirs(AGENT_TASKS)
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}

    if not tasks:
        print("ERROR: no tasks found", file=sys.stderr)
        sys.exit(1)

    print("Cleaning up leftover network state...", flush=True)
    subprocess.run(
        "sudo ip netns list 2>/dev/null | grep -E 'fb_(A|QT|QW)' | "
        "while read ns rest; do "
        "ns_name=$(echo \"$ns\" | tr -d ':'); "
        "sudo ip netns del \"$ns_name\" 2>/dev/null; done; "
        "ip link show 2>/dev/null | grep -oP 'veth_(A|QT|QW)[^ @]+' | "
        "while read v; do "
        "sudo ip link del \"$v\" 2>/dev/null; done; "
        "sudo rm -rf /etc/netns/fb_A* /etc/netns/fb_QT* /etc/netns/fb_QW* 2>/dev/null",
        shell=True, capture_output=True, timeout=15,
    )
    # Kill leftover mock servers (only when NOT using external mock)
    if not args.no_mock:
        for proc in Path("/proc").iterdir():
            try:
                pid = int(proc.name)
                cmdline = (proc / "cmdline").read_text()
                if "mock_" in cmdline and "_server.py" in cmdline:
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Killed leftover mock server pid={pid}", flush=True)
            except (ValueError, OSError, PermissionError):
                continue
    print("  Done.\n", flush=True)

    count_per_task = args.count if args.count is not None else SAMPLES_PER_TASK
    if count_per_task == 0:
        count_per_task = None  # unlimited
    total_ep = len(tasks) * (count_per_task if count_per_task else SAMPLES_PER_TASK)

    print(f"=== Qwen batch: {len(tasks)} tasks, ~{total_ep} episodes ===")
    print(f"Model: {args.model} | Timeout: {args.timeout}s")
    if args.dry_run:
        print("[DRY RUN MODE]")
    if args.supplement:
        print("[SUPPLEMENT MODE — skip already-successful episodes]")
    print()

    results = {}
    total_count = 0
    t0 = time.time()

    for task_num in sorted(tasks):
        task_dir = tasks[task_num]
        limit = count_per_task
        instances = find_instance_files(task_dir, limit=limit)

        print(f"-- {task_dir.name} ({len(instances)} samples) --", flush=True)

        task_rc = []
        skipped = 0
        for idx, inst in enumerate(instances):
            total_count += 1
            eid = f"A{task_num:02d}{idx:02d}"

            if args.supplement and episode_already_done(eid):
                print(f"  [{eid}] already done, skipping", flush=True)
                task_rc.append(0)
                skipped += 1
                continue

            clean_netns(eid)
            rc = run_one_qwen_episode(eid, task_dir, inst,
                                      args.model, args.timeout,
                                      args.dry_run, args.no_mock,
                                      args.no_docker)
            task_rc.append(rc)

            if not args.dry_run:
                clean_netns(eid)
                time.sleep(1)
                prune_episode(eid, rc)

        ok = sum(1 for r in task_rc if r == 0)
        timeout_n = sum(1 for r in task_rc if r == -1)
        fail = len(task_rc) - ok - timeout_n
        ski = f" SKIP={skipped}" if skipped else ""
        print(f"  => OK={ok} FAIL={fail} TIMEOUT={timeout_n}{ski}\n", flush=True)
        results[task_num] = task_rc

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  QWEN BATCH DONE: {total_count} episodes in {elapsed/60:.0f} min")
    print(f"{'='*60}\n")
    for num in sorted(results):
        rc_list = results[num]
        ok = sum(1 for r in rc_list if r == 0)
        print(f"  {tasks[num].name:40s} OK={ok}/{len(rc_list)}")
    print(f"\nReport: episodes/Agent/qwen/qwen_batch_report.json")

    report = {
        "mode": "qwen",
        "episodes": total_count, "total_s": elapsed,
        "model": args.model, "timeout_s": args.timeout,
        "tasks": {
            str(n): {"dir": tasks[n].name,
                     "ok": sum(1 for r in rc_list if r == 0),
                     "results": rc_list}
            for n, rc_list in results.items()
        },
    }
    with open(EPISODES_DIR / "qwen_batch_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
