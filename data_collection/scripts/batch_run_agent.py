#!/usr/bin/env python3
"""Batch run all AGENT FlowBenchmark tasks (015-021), 5 samples each.

Each sample = independent episode with isolated netns.
Agent tasks use CLI harness (Claude Code / Codex / Gemini CLI).

Usage:
    python scripts/batch_run_agent.py [--dry-run] [--task N] [--harness claude_code_deepseek] [--timeout 1800]
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
EPISODES_DIR = PROJECT_ROOT / "episodes" / "Agent" / "cc_dpsk"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNNER_DIR = SCRIPTS_DIR / "runner"
ENV_DIR = SCRIPTS_DIR / "env"

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

SAMPLES_PER_TASK = 10
EXIT_IDLE_KILLED = 241   # SIGTERM from idle detection in run_cli_task.py
SKIP_TASKS = {19,22}

# Agent/CLI tasks handled by this serial batch runner.
AGENT_TASKS = {15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 40}


DEFAULT_HARNESS = "claude_code_deepseek"
DEFAULT_TIMEOUT = 3600  # 60 min per agent episode

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
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=15,
    )

def kill_netns(episode_id):
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=30,
    )

def run_one_agent_episode(episode_id, task_dir, instance_path,
                          harness, timeout_s, dry_run=False,
                          no_mock=False):
    cmd = [
        sys.executable, str(RUNNER_DIR / "run_episode.py"), episode_id,
        "--runner", "cli",
        "--instance", str(instance_path),
        "--harness", harness,
        "--episodes-subdir", "Agent/cc_dpsk",
        "--timeout", str(timeout_s),
    ]
    if no_mock:
        cmd.append("--no-mock")

    print(f"  [{episode_id}] {task_dir.name}/{instance_path.name} "
          f"harness={harness}", flush=True)

    if dry_run:
        return 0

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(ep_dir)],
                   capture_output=True, timeout=5)

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
            kill_netns(episode_id)
            _child_proc.terminate()
            try:
                _child_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _child_proc.kill()
            rc = -1
            status = "TIMEOUT"
        else:
            if rc == EXIT_IDLE_KILLED:
                status = "IDLE_KILLED"
            else:
                status = "OK" if rc == 0 else f"FAIL({rc})"
    finally:
        _child_proc = None

    elapsed = time.time() - t0

    meta = {
        "episode_id": episode_id, "task": task_dir.name,
        "instance": str(instance_path), "exit_code": rc,
        "elapsed_s": elapsed, "timeout_s": timeout_s,
        "harness": harness,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    -> {status} ({elapsed:.0f}s)", flush=True)
    return rc

def prune_episode(episode_id, rc):
    """On success, remove debug logs to save space.
    For agent tasks, also verify cli_response.json exists."""
    ep_dir = EPISODES_DIR / episode_id
    really_ok = (rc == 0 and (ep_dir / "cli_response.json").exists())
    if not really_ok:
        return
    for fname in ("orchestrator_stdout.log", "orchestrator_stderr.log",
                  "stderr.log", "stdout.log"):
        (ep_dir / fname).unlink(missing_ok=True)

def episode_already_done(episode_id):
    """Check if an agent episode already completed successfully."""
    resp = EPISODES_DIR / episode_id / "cli_response.json"
    if not resp.exists():
        return False
    try:
        d = json.loads(resp.read_text())
        return d.get("completed", False)
    except Exception:
        return False

def episode_should_skip(episode_id):
    """Check if episode should be skipped in supplement mode.
    Skips: successfully completed, or killed by idle detection (not a real failure)."""
    if episode_already_done(episode_id):
        return True
    # Killed by idle detection — not a real failure, skip re-run
    batch_meta = EPISODES_DIR / episode_id / "batch_meta.json"
    if batch_meta.exists():
        try:
            d = json.loads(batch_meta.read_text())
            if d.get("exit_code") == EXIT_IDLE_KILLED:
                return True
        except Exception:
            pass
    return False

def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    parser = argparse.ArgumentParser(description="Batch run all AGENT episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--harness", default=DEFAULT_HARNESS,
                        help=f"CLI harness (default {DEFAULT_HARNESS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Seconds per episode (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--count", type=int, default=None,
                        help=f"Max samples per task (default: {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--no-mock", action="store_true",
                        help="Skip starting mock services (use if already running)")
    parser.add_argument("--supplement", action="store_true",
                        help="Skip already-successful episodes, only re-run failed/missing")
    parser.add_argument("--skip-initial-cleanup", action="store_true",
                        help="Skip global fb_A* cleanup; used by task-level parallel launcher")
    parser.add_argument("--keep-logs", action="store_true",
                        help="Keep stdout/stderr logs even on success")
    args = parser.parse_args()

    tasks = find_task_dirs(AGENT_TASKS)
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}

    if args.skip_initial_cleanup:
        print("Skipping initial global cleanup (managed by parallel launcher).\n", flush=True)
    else:
        # Only clean up netns/veth/DNS for the task(s) being run in this batch.
        print("Cleaning up leftover network state for selected tasks...", flush=True)
        for task_num in sorted(tasks):
            prefix = f"A{task_num:02d}"
            subprocess.run(
                f"sudo ip netns list 2>/dev/null | grep 'fb_{prefix}' | while read ns; do "
                f"sudo ip netns del \"$ns\" 2>/dev/null; done; "
                f"ip link show 2>/dev/null | grep -oP 'veth_{prefix}[^ @]+' | while read v; do "
                f"sudo ip link del \"$v\" 2>/dev/null; done",
                shell=True, capture_output=True, timeout=15,
            )
            subprocess.run(
                ["sudo", "rm", "-rf", f"/etc/netns/fb_{prefix}*"],
                capture_output=True, timeout=5,
            )
        print("  Done.\n", flush=True)

    limit_for_total = None if args.count == 0 else args.count
    total_ep = sum(len(find_instance_files(task_dir, limit=limit_for_total))
                   for task_dir in tasks.values())
    print(f"=== AGENT batch: {len(tasks)} tasks, {total_ep} episodes ===")
    print(f"Harness: {args.harness} | Timeout: {args.timeout}s")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    results = {}
    count = 0
    t0 = time.time()

    for task_num in sorted(tasks):
        task_dir = tasks[task_num]
        limit = None if args.count == 0 else args.count  # None→default, 0→all, N→N
        instances = find_instance_files(task_dir, limit=limit)

        print(f"-- {task_dir.name} ({len(instances)} samples) --", flush=True)

        task_rc = []
        skipped = 0
        for idx, inst in enumerate(instances):
            count += 1
            eid = f"A{task_num:02d}{idx:02d}"

            if args.supplement and episode_should_skip(eid):
                print(f"  [{eid}] already done, skipping", flush=True)
                task_rc.append(0)
                skipped += 1
                continue

            clean_netns(eid)
            rc = run_one_agent_episode(eid, task_dir, inst,
                                       args.harness, args.timeout,
                                       args.dry_run, args.no_mock)

            task_rc.append(rc)

            if not args.dry_run:
                clean_netns(eid)
                time.sleep(1)
                if not args.keep_logs:
                    prune_episode(eid, rc)

        ok = sum(1 for r in task_rc if r == 0)
        timeout_n = sum(1 for r in task_rc if r == -1)
        idle_killed = sum(1 for r in task_rc if r == EXIT_IDLE_KILLED)
        fail = len(task_rc) - ok - timeout_n - idle_killed
        ski = f" SKIP={skipped}" if skipped else ""
        parts = [f"OK={ok}", f"FAIL={fail}"]
        if idle_killed:
            parts.append(f"IDLE_KILLED={idle_killed}")
        if timeout_n:
            parts.append(f"TIMEOUT={timeout_n}")
        print(f"  => {' '.join(parts)}{ski}\n", flush=True)
        results[task_num] = task_rc

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  AGENT BATCH DONE: {count} episodes in {elapsed/60:.0f} min")
    print(f"{'='*60}\n")
    for num in sorted(results):
        rc_list = results[num]
        ok = sum(1 for r in rc_list if r == 0)
        print(f"  {tasks[num].name:40s} OK={ok}/{len(rc_list)}")
    print(f"\nReport: episodes/agent_batch_report.json")

    report = {
        "mode": "agent",
        "episodes": count, "total_s": elapsed,
        "harness": args.harness, "timeout_s": args.timeout,
        "tasks": {
            str(n): {"dir": tasks[n].name,
                     "ok": sum(1 for r in rc_list if r == 0),
                     "fail": sum(1 for r in rc_list if r not in (0, -1, EXIT_IDLE_KILLED)),
                     "idle_killed": sum(1 for r in rc_list if r == EXIT_IDLE_KILLED),
                     "timeout": sum(1 for r in rc_list if r == -1),
                     "results": rc_list}
            for n, rc_list in results.items()
        },
    }
    with open(EPISODES_DIR / "agent_batch_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
