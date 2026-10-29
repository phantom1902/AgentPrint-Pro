#!/usr/bin/env python3
"""Batch run all WEB FlowBenchmark tasks (001-014, 023-032), 5 samples each.

Each sample = independent episode with isolated netns.
Episode timeout: if exceeded, kill and move to next.
After collection: label traffic + generate Gantt chart.

Usage:
    python scripts/batch_run_web.py [--dry-run] [--task N] [--platform qwen] [--model qwen3-max] [--timeout 120]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
EPISODES_DIR = PROJECT_ROOT / "episodes"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNNER_DIR = SCRIPTS_DIR / "runner"
ENV_DIR = SCRIPTS_DIR / "env"

SAMPLES_PER_TASK = 20
SKIP_TASKS = {2,22}

# Web-only tasks: 001-014 are pure web, 023-032 are web/agent (default web)
# WEB_TASKS = set(range(1, 15)) | set(range(23, 33))
# WEB_TASKS = {1,3,4,5,6,7,8,9,10,11,23,24,25,26,27,28,29,30,31,32}
WEB_TASKS = {10,11,23,24,25,26,27,28,29,30,31,32}

DEFAULT_PLATFORM = "qwen"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_TIMEOUT = 600  # seconds per episode (10 min, enough for image gen)


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
    """Force-kill all processes in namespace and delete it."""
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=30,
    )


def run_one_web_episode(episode_id, task_dir, instance_path,
                        platform, model, timeout_s, dry_run=False,
                        no_mock=False):
    cmd = [
        sys.executable, str(RUNNER_DIR / "run_episode.py"), episode_id,
        "--runner", "web",
        "--instance", str(instance_path),
        "--platform", platform,
        "--model", model,
        "--timeout", str(timeout_s),
    ]
    if no_mock:
        cmd.append("--no-mock")

    print(f"  [{episode_id}] {task_dir.name}/{instance_path.name}", flush=True)

    if dry_run:
        return 0

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    # netns creation runs as root and may change ownership; fix it
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
            rc = _child_proc.wait(timeout=timeout_s + 30)
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
            status = "OK" if rc == 0 else f"FAIL({rc})"
    finally:
        _child_proc = None

    elapsed = time.time() - t0

    meta = {
        "episode_id": episode_id, "task": task_dir.name,
        "instance": str(instance_path), "exit_code": rc,
        "elapsed_s": elapsed, "timeout_s": timeout_s,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    -> {status} ({elapsed:.0f}s)", flush=True)
    return rc


def prune_episode(episode_id, rc):
    """On success, remove debug logs to save space.
    For web tasks, also verify web_response.json exists (rc=0 from orchestrator
    doesn't mean the web task actually completed)."""
    ep_dir = EPISODES_DIR / episode_id
    really_ok = (rc == 0 and (ep_dir / "web_response.json").exists())
    if not really_ok:
        return
    for fname in ("orchestrator_stdout.log", "orchestrator_stderr.log",
                  "stderr.log", "stdout.log"):
        (ep_dir / fname).unlink(missing_ok=True)


def episode_already_done(episode_id):
    """Check if a web episode already completed successfully."""
    resp = EPISODES_DIR / episode_id / "web_response.json"
    if not resp.exists():
        return False
    try:
        d = json.loads(resp.read_text())
        return bool(d.get("response", "").strip())
    except Exception:
        return False


def label_one_episode(episode_id):
    ep_dir = EPISODES_DIR / episode_id
    pcap = ep_dir / "network.pcap"
    events = ep_dir / "events.jsonl"
    if not pcap.exists() or not events.exists():
        return
    # Run labeling and charting in parallel, then wait for both to finish
    # before the next episode starts.  This prevents memory from
    # accumulating across episodes when Chromium is launched again.
    p1 = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "label_pcap.py"), str(ep_dir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    p2 = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "visualize_episode.py"), str(ep_dir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for p in (p1, p2):
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    parser = argparse.ArgumentParser(description="Batch run all WEB episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Seconds per episode (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--count", type=int, default=None,
                        help=f"Max samples per task (default: {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--no-mock", action="store_true",
                        help="Skip starting mock services (use if already running)")
    parser.add_argument("--supplement", action="store_true",
                        help="Skip already-successful episodes, only re-run failed/missing")
    args = parser.parse_args()

    tasks = find_task_dirs(WEB_TASKS)
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}

    # Clean up leftover netns/veth from previous incomplete runs (W-prefix only).
    print("Cleaning up leftover network state...", flush=True)
    subprocess.run(
        "sudo ip netns list 2>/dev/null | grep 'fb_W' | while read ns; do "
        "sudo ip netns del \"$ns\" 2>/dev/null; done; "
        "ip link show 2>/dev/null | grep -oP 'veth_W[^ @]+' | while read v; do "
        "sudo ip link del \"$v\" 2>/dev/null; done; "
        "sudo rm -rf /etc/netns/fb_W* 2>/dev/null",
        shell=True, capture_output=True, timeout=15,
    )
    print("  Done.\n", flush=True)

    total_ep = len(tasks) * SAMPLES_PER_TASK
    print(f"=== WEB batch: {len(tasks)} tasks, {total_ep} episodes ===")
    print(f"Platform: {args.platform} | Model: {args.model} | Timeout: {args.timeout}s")
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
            eid = f"W{task_num:02d}{idx:02d}"

            if args.supplement and episode_already_done(eid):
                print(f"  [{eid}] already done, skipping", flush=True)
                task_rc.append(0)
                skipped += 1
                continue

            clean_netns(eid)
            rc = run_one_web_episode(eid, task_dir, inst,
                                     args.platform, args.model, args.timeout,
                                     args.dry_run, args.no_mock)

            # Auto-retry once on transient failures (skip if session expired)
            if rc != 0 and not args.dry_run \
               and not (EPISODES_DIR / eid / "session_expired.flag").exists():
                clean_netns(eid)
                time.sleep(5)
                print(f"  [{eid}] retrying after failure (rc={rc})...", flush=True)
                rc = run_one_web_episode(eid, task_dir, inst,
                                         args.platform, args.model, args.timeout,
                                         args.dry_run, args.no_mock)

            task_rc.append(rc)

            if not args.dry_run:
                clean_netns(eid)
                time.sleep(0.5)
                if not args.skip_label:
                    label_one_episode(eid)
                prune_episode(eid, rc)

                # Stop batch immediately on session expiry
                if (EPISODES_DIR / eid / "session_expired.flag").exists():
                    print(f"\n  *** SESSION EXPIRED — stopping batch. Please re-login: ***")
                    print(f"  python scripts/runner/init_web_profile.py --platform {args.platform}")
                    print(f"  Then re-run: python scripts/batch_run_web.py --supplement --no-mock\n")
                    sys.exit(1)

        # Per-task summary
        ok = sum(1 for r in task_rc if r == 0)
        timeout_n = sum(1 for r in task_rc if r == -1)
        fail = len(task_rc) - ok - timeout_n
        ski = f" SKIP={skipped}" if skipped else ""
        print(f"  => OK={ok} FAIL={fail} TIMEOUT={timeout_n}{ski}\n", flush=True)
        results[task_num] = task_rc

    elapsed = time.time() - t0

    # Final summary
    print(f"\n{'='*60}")
    print(f"  WEB BATCH DONE: {count} episodes in {elapsed/60:.0f} min")
    print(f"{'='*60}\n")
    for num in sorted(results):
        rc_list = results[num]
        ok = sum(1 for r in rc_list if r == 0)
        fail = len(rc_list) - ok
        print(f"  {tasks[num].name:40s} OK={ok}/{len(rc_list)}")
    print(f"\nReport: episodes/web_batch_report.json")

    report = {
        "mode": "web",
        "episodes": count, "total_s": elapsed,
        "platform": args.platform, "model": args.model,
        "timeout_s": args.timeout,
        "tasks": {
            str(n): {"dir": tasks[n].name,
                     "ok": sum(1 for r in rc_list if r == 0),
                     "results": rc_list}
            for n, rc_list in results.items()
        },
    }
    with open(EPISODES_DIR / "web_batch_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
