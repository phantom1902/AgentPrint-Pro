#!/usr/bin/env python3
"""Batch run FlowBenchmark agent tasks with Codex CLI.

Runs samples sequentially and stores outputs under episodes/codex/Axxxx.
Supports restricting by task/sample range and supplement mode that only runs
missing or unsuccessful episodes.
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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TASKS_DIR = PROJECT_ROOT / "tasks"
EPISODES_DIR = PROJECT_ROOT / "episodes" / "Agent" / "codex"
RUN_EPISODE_PATH = SCRIPTS_DIR / "runner" / "run_episode.py"
ENV_DIR = SCRIPTS_DIR / "env"

SAMPLES_PER_TASK = 10
SKIP_TASKS = {19, 22}
AGENT_TASKS = {15,16,17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 40}
DEFAULT_TIMEOUT = 1800

_child_proc = None
_current_episode = None
CLEANUP_SCRIPT = str(PROJECT_ROOT / "scripts" / "env" / "cleanup_episode_netns.sh")

def _handle_sigint(sig, frame):
    del sig, frame
    global _child_proc, _current_episode
    if _child_proc and _child_proc.poll() is None:
        _child_proc.terminate()
        try:
            _child_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _child_proc.kill()
    # Kill everything inside the netns — run_episode spawns codex via
    # sudo + bwrap, which won't be reached by killing just its parent.
    if _current_episode:
        subprocess.run(
            ["sudo", CLEANUP_SCRIPT, _current_episode],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
        )
    print("\nInterrupted.", flush=True)
    sys.exit(130)

def parse_range(value: str, default_start: int, default_end: int) -> tuple[int, int]:
    text = (value or "").strip()
    if not text:
        return default_start, default_end
    if "-" not in text:
        point = int(text)
        return point, point
    left, right = text.split("-", 1)
    start = int(left) if left else default_start
    end = int(right) if right else default_end
    if start > end:
        raise ValueError(f"Invalid range: {value}")
    return start, end

def find_task_dirs() -> dict[int, Path]:
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
        if num in SKIP_TASKS or num not in AGENT_TASKS:
            continue
        tasks[num] = d
    return tasks

def find_instance_files(task_dir: Path) -> list[Path]:
    inst_dir = task_dir / "instances"
    if not inst_dir.exists():
        return []
    return sorted(inst_dir.glob("sample_*.yaml"))

def episode_id_for(task_num: int, sample_num: int) -> str:
    return f"A{task_num:02d}{sample_num:02d}"

def clean_netns(episode_id: str):
    subprocess.run(
        ["sudo", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=15,
    )

def episode_completed(episode_id: str) -> bool:
    resp = EPISODES_DIR / episode_id / "cli_response.json"
    if not resp.exists():
        return False
    try:
        data = json.loads(resp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("completed")) and int(data.get("exit_code", 1)) == 0

def run_one_episode(
    episode_id: str,
    task_dir: Path,
    instance_path: Path,
    model: str,
    timeout_s: int,
    dry_run: bool,
    enable_search: bool,
    sandbox: str,
    approval_policy: str,
    danger_full_access: bool,
    extra_codex_args: list[str],
    no_mock: bool = False,
):
    cmd = [
        sys.executable,
        str(RUN_EPISODE_PATH),
        episode_id,
        "--runner",
        "cli",
        "--instance",
        str(instance_path),
        "--harness",
        "codex",
        "--episodes-subdir",
        "Agent/codex",
        "--timeout",
        str(timeout_s),
        "--codex-sandbox",
        sandbox,
        "--codex-approval-policy",
        approval_policy,
    ]
    if model:
        cmd.extend(["--model", model])
    if enable_search:
        cmd.append("--codex-enable-search")
    if danger_full_access:
        cmd.append("--codex-danger-full-access")
    if no_mock:
        cmd.append("--no-mock")
    for arg in extra_codex_args:
        cmd.extend(["--codex-arg", arg])

    print(f"  [{episode_id}] {task_dir.name}/{instance_path.name} model={model or '(config default)'}", flush=True)

    if dry_run:
        return 0

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(ep_dir / "orchestrator_stdout.log", "w", encoding="utf-8")
    stderr_handle = open(ep_dir / "orchestrator_stderr.log", "w", encoding="utf-8")

    global _child_proc, _current_episode
    _current_episode = episode_id
    t0 = time.time()
    try:
        _child_proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        try:
            rc = _child_proc.wait(timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            _child_proc.terminate()
            try:
                _child_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _child_proc.kill()
                _child_proc.wait(timeout=10)
            rc = -1
    finally:
        _child_proc = None
        _current_episode = None
        stdout_handle.close()
        stderr_handle.close()

    elapsed = time.time() - t0
    meta = {
        "episode_id": episode_id,
        "task": task_dir.name,
        "instance": str(instance_path),
        "exit_code": rc,
        "elapsed_s": elapsed,
        "timeout_s": timeout_s,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    status = "OK" if rc == 0 else ("TIMEOUT" if rc == -1 else f"FAIL({rc})")
    print(f"    -> {status} ({elapsed:.0f}s)", flush=True)
    return rc

def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Batch run Codex agent episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="", help="Codex model override")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--task", type=int, default=0, help="Single task id")
    parser.add_argument("--task-range", default="", help="Task range, e.g. 23-25")
    parser.add_argument("--sample-range", default="", help="Sample index range, e.g. 0-9")
    parser.add_argument("--count", type=int, default=None, help=f"Max samples per task (default {SAMPLES_PER_TASK}, 0=all)")
    parser.add_argument("--supplement", action="store_true", help="Only run missing or unsuccessful episodes")
    parser.add_argument("--enable-search", dest="enable_search", action="store_true", help="Pass --enable-search to Codex")
    parser.add_argument("--disable-search", dest="enable_search", action="store_false", help="Pass --disable-search behavior to Codex")
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
    )
    parser.add_argument(
        "--approval-policy",
        default="never",
        choices=["untrusted", "on-failure", "on-request", "never"],
    )
    parser.add_argument("--danger-full-access", action="store_true")
    parser.add_argument("--no-mock", action="store_true", help="Skip starting mock services (use if already running)")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra arg passed through to test_codex_agent_task.py")
    parser.set_defaults(enable_search=True)
    args = parser.parse_args()

    all_tasks = find_task_dirs()
    if args.task:
        tasks = {args.task: all_tasks[args.task]} if args.task in all_tasks else {}
    else:
        task_start, task_end = parse_range(args.task_range, min(all_tasks), max(all_tasks))
        tasks = {num: path for num, path in all_tasks.items() if task_start <= num <= task_end}

    sample_start, sample_end = parse_range(args.sample_range, 0, 99)

    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cleaning up leftover network state + mock services...", flush=True)
    subprocess.run(
        "sudo ip netns list 2>/dev/null | grep 'fb_A' | while read ns; do "
        "sudo ip netns del \"$ns\" 2>/dev/null; done; "
        "ip link show 2>/dev/null | grep -oP 'veth_A[^ @]+' | while read v; do "
        "sudo ip link del \"$v\" 2>/dev/null; done; "
        "sudo rm -rf /etc/netns/fb_A* 2>/dev/null",
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

    print(f"=== CODEX batch: {len(tasks)} tasks ===")
    print(f"Model: {args.model or '(config default)'} | Timeout: {args.timeout}s | Episodes root: {EPISODES_DIR}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    results = {}
    total = 0
    t0 = time.time()

    for task_num in sorted(tasks):
        task_dir = tasks[task_num]
        instances = find_instance_files(task_dir)
        if args.count is None:
            instances = instances[:SAMPLES_PER_TASK]
        elif args.count > 0:
            instances = instances[:args.count]

        selected = []
        for idx, inst in enumerate(instances):
            if sample_start <= idx <= sample_end:
                selected.append((idx, inst))

        print(f"-- {task_dir.name} ({len(selected)} samples selected) --", flush=True)
        task_rc = []
        skipped = 0

        for idx, inst in selected:
            total += 1
            eid = episode_id_for(task_num, idx)
            if args.supplement and episode_completed(eid):
                print(f"  [{eid}] already done, skipping", flush=True)
                task_rc.append(0)
                skipped += 1
                continue

            clean_netns(eid)
            rc = run_one_episode(
                eid,
                task_dir,
                inst,
                args.model,
                args.timeout,
                args.dry_run,
                args.enable_search,
                args.sandbox,
                args.approval_policy,
                args.danger_full_access,
                args.codex_arg,
                no_mock=args.no_mock,
            )
            task_rc.append(rc)

            if not args.dry_run:
                clean_netns(eid)
                time.sleep(1)

        ok = sum(1 for r in task_rc if r == 0)
        timeout_n = sum(1 for r in task_rc if r == -1)
        fail = len(task_rc) - ok - timeout_n
        extra = f" SKIP={skipped}" if skipped else ""
        print(f"  => OK={ok} FAIL={fail} TIMEOUT={timeout_n}{extra}\n", flush=True)
        results[task_num] = task_rc

    elapsed = time.time() - t0
    report = {
        "mode": "codex",
        "episodes": total,
        "total_s": elapsed,
        "model": args.model,
        "timeout_s": args.timeout,
        "task_range": args.task_range,
        "sample_range": args.sample_range,
        "supplement": args.supplement,
        "tasks": {
            str(num): {
                "dir": tasks[num].name,
                "ok": sum(1 for r in rc_list if r == 0),
                "results": rc_list,
            }
            for num, rc_list in results.items()
        },
    }
    report_path = EPISODES_DIR / "codex_batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"  CODEX BATCH DONE: {total} episodes in {elapsed/60:.0f} min")
    print("=" * 60)
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
