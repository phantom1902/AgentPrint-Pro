#!/usr/bin/env python3
"""
Batch run Task 36 (Spear Phishing) episodes sequentially.

Each episode runs an attacker agent (LLM-driven) against a target agent (Codex)
in an isolated network namespace, with tcpdump capturing all traffic on the veth.

Output goes to episodes/Agent/<harness_subdir>/ matching the same pattern as all
other FlowBenchmark tasks.

Usage:
    # Run first 10 samples with default harness
    python scripts/batch_run_spear_phishing.py --count 10

    # Run specific samples with custom harness/model
    python scripts/batch_run_spear_phishing.py --count 30 \\
        --harness claude_code_deepseek --model deepseek-v4-flash

    # Dry run
    python scripts/batch_run_spear_phishing.py --count 5 --dry-run
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
TASK_DIR = PROJECT_ROOT / "tasks" / "036_spear_phishing"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNNER_DIR = SCRIPTS_DIR / "runner"
ENV_DIR = SCRIPTS_DIR / "env"

# ── Harness → episodes subdirectory mapping ──
# Must match the directory names under episodes/Agent/
HARNESS_TO_SUBDIR = {
    "claude_code_claude":       "cc_claude",
    "claude_code_deepseek":     "cc_dpsk",
    "claude_code_mimo-v2.5pro": "cc_mimo",
    "codex_openai":             "codex",
    "codex_deepseek":           "codex",
    "qwen_code_qwen":           "qwen",
    "qwen_code_qwen36":         "qwen",
    "kimi_code_kimi":           "kimi",
}


def get_subdir(harness: str) -> str:
    """Map harness preset name to episodes subdirectory."""
    return HARNESS_TO_SUBDIR.get(harness, harness)


def get_episodes_dir(harness: str) -> Path:
    """Return the episodes output directory for a given harness."""
    subdir = get_subdir(harness)
    return PROJECT_ROOT / "episodes" / "Agent" / subdir


_child_proc = None
_stop_requested = False


def _handle_sigint(sig, frame):
    """Graceful shutdown: stop after the current episode finishes, don't abort mid-run."""
    global _stop_requested, _child_proc
    sig_name = signal.Signals(sig).name
    if _stop_requested:
        # Second signal — force kill
        print(f"\nSecond {sig_name} received. Force killing...", flush=True)
        if _child_proc and _child_proc.poll() is None:
            _child_proc.kill()
        sys.exit(128 + sig)
    _stop_requested = True
    print(f"\n{signal.Signals(sig).name} received. Will stop after current episode finishes."
          " Press Ctrl+C again to force quit.", flush=True)
    # Terminate current child so it doesn't keep running for the full timeout
    if _child_proc and _child_proc.poll() is None:
        _child_proc.terminate()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

DEFAULT_COUNT = 10
DEFAULT_HARNESS = "claude_code_deepseek"
DEFAULT_TIMEOUT = 1800
EXIT_IDLE_KILLED = 241


def find_instance_files(start: int = 0, count: int = DEFAULT_COUNT):
    inst_dir = TASK_DIR / "instances"
    if not inst_dir.exists():
        print(f"ERROR: instances directory not found: {inst_dir}", file=sys.stderr)
        return []
    all_instances = sorted(inst_dir.glob("sample_*.yaml"))
    return all_instances[start:start + count]


def force_clean_netns(episode_id: str):
    """Aggressively clean up any leftover namespace, veth, and iptables rules."""
    subprocess.run(
        ["sudo", "bash", str(ENV_DIR / "force_cleanup_netns.sh"), episode_id],
        capture_output=True, timeout=30,
    )


def clean_netns(episode_id: str):
    """Gentle cleanup via the standard script."""
    subprocess.run(
        ["sudo", "bash", str(ENV_DIR / "cleanup_episode_netns.sh"), episode_id],
        capture_output=True, timeout=30,
    )


def run_one_episode(episode_id: str, instance_path: Path,
                    harness: str, model: str, timeout_s: int,
                    episodes_subdir: str,
                    dry_run: bool = False) -> int:
    """Run a single spear phishing episode via run_episode.py."""
    episodes_dir = get_episodes_dir(harness)

    cmd = [
        sys.executable, str(RUNNER_DIR / "run_episode.py"), episode_id,
        "--runner", "multi_agent",
        "--instance", str(instance_path),
        "--harness", harness,
        "--episodes-subdir", f"Agent/{episodes_subdir}",
        "--timeout", str(timeout_s),
    ]
    if model:
        cmd += ["--model", model]

    inst_name = instance_path.stem
    print(f"  [{episode_id}] {inst_name}  harness={harness}"
          f"{' model=' + model if model else ''}", flush=True)

    if dry_run:
        print("    [DRY RUN] skip", flush=True)
        return 0

    ep_dir = episodes_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Fix ownership
    subprocess.run(
        ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(ep_dir)],
        capture_output=True, timeout=5,
    )

    t0 = time.time()
    rc = -1
    status = "UNKNOWN"

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
            force_clean_netns(episode_id)
            _child_proc.terminate()
            try:
                _child_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _child_proc.kill()
                _child_proc.wait(timeout=5)
            rc = -1
            status = "TIMEOUT"
        else:
            if rc == EXIT_IDLE_KILLED:
                status = "IDLE_KILLED"
            elif rc == 0:
                status = "OK"
            else:
                status = f"FAIL({rc})"
    finally:
        _child_proc = None

    elapsed = time.time() - t0

    meta = {
        "episode_id": episode_id,
        "task": "036_spear_phishing",
        "instance": str(instance_path),
        "exit_code": rc,
        "elapsed_s": elapsed,
        "timeout_s": timeout_s,
        "harness": harness,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        with open(ep_dir / "batch_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    print(f"    -> {status} ({elapsed:.0f}s)", flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description="Batch run Task 36 Spear Phishing episodes")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of episodes (default: {DEFAULT_COUNT})")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index, 0-based (default: 0)")
    parser.add_argument("--harness", default=DEFAULT_HARNESS,
                        help=f"Attacker harness preset (default: {DEFAULT_HARNESS})")
    parser.add_argument("--model", default="",
                        help="Model override (default: derived from harness preset)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-episode timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    instances = find_instance_files(args.start, args.count)
    if not instances:
        print(f"ERROR: No instances found (start={args.start}, count={args.count})",
              file=sys.stderr)
        sys.exit(1)

    episodes_subdir = get_subdir(args.harness)
    episodes_dir = get_episodes_dir(args.harness)

    total = len(instances)
    est_minutes = total * args.timeout // 60

    print(f"=== Task 36 Spear Phishing Batch Run ===")
    print(f"  Instances:   {total} (indices {args.start}–{args.start + total - 1})")
    print(f"  Harness:     {args.harness}")
    if args.model:
        print(f"  Model:       {args.model}")
    print(f"  Timeout:     {args.timeout}s per episode")
    print(f"  Est. total:  ~{est_minutes} min")
    print(f"  Output:      {episodes_dir}")
    if args.dry_run:
        print(f"  [DRY RUN MODE]")
    print()

    results = {}
    t_start = time.time()

    for idx, inst_path in enumerate(instances):
        sample_num = args.start + idx
        # Episode ID must be ≤5 chars so veth_<ID>_host ≤ 15 chars (IFNAMSIZ limit).
        # Format: 36000–36299 (supports all 300 samples).
        episode_id = f"{36:02d}{sample_num:03d}"

        # Force-clean any leftovers from previous runs before starting
        print(f"  Cleaning up {episode_id}...", flush=True)
        force_clean_netns(episode_id)
        time.sleep(1)

        rc = run_one_episode(
            episode_id=episode_id,
            instance_path=inst_path,
            harness=args.harness,
            model=args.model,
            timeout_s=args.timeout,
            episodes_subdir=episodes_subdir,
            dry_run=args.dry_run,
        )

        results[sample_num] = {
            "episode_id": episode_id,
            "instance": inst_path.name,
            "exit_code": rc,
        }

        if not args.dry_run:
            clean_netns(episode_id)
            time.sleep(2)

        # Check if stop was requested between episodes
        if _stop_requested:
            print(f"\n  Stop requested after {idx + 1} episodes. Exiting.", flush=True)
            break

    elapsed_total = time.time() - t_start

    # ── Summary ──
    ok = sum(1 for r in results.values() if r["exit_code"] == 0)
    fail = sum(1 for r in results.values() if r["exit_code"] not in (0,))
    timeout_n = sum(1 for r in results.values() if r["exit_code"] == -1)

    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE: {total} episodes in {elapsed_total/60:.0f} min")
    print(f"  OK: {ok}  FAIL: {fail}  TIMEOUT: {timeout_n}")
    print(f"{'='*60}")

    if fail > 0:
        print(f"\n  Failed episodes:")
        for idx, r in results.items():
            if r["exit_code"] != 0:
                eid = f"{36:02d}{idx:03d}"
                print(f"    {eid}  {r['instance']}  rc={r['exit_code']}")

    report = {
        "task": "036_spear_phishing",
        "harness": args.harness,
        "model": args.model or "default",
        "total": total, "ok": ok, "fail": fail, "timeout": timeout_n,
        "elapsed_s": elapsed_total, "timeout_s": args.timeout,
        "results": results,
    }
    report_path = episodes_dir / "batch_report_spear_phishing.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report: {report_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
