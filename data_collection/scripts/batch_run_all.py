#!/usr/bin/env python3
"""Batch run all FlowBenchmark tasks (except 022), 5 samples each.

Each sample = independent episode with isolated netns.
After collection: label traffic + generate Gantt chart.

Usage:
    python scripts/batch_run_all.py [--dry-run] [--task N] [--platform qwen] [--model qwen3-max]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "tasks"
EPISODES_DIR = PROJECT_ROOT / "episodes"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNNER_DIR = SCRIPTS_DIR / "runner"
ENV_DIR = SCRIPTS_DIR / "env"

SAMPLES_PER_TASK = 5
SKIP_TASKS = {22}
DEFAULT_PLATFORM = "qwen"
DEFAULT_MODEL = "qwen3-max"
DEFAULT_HARNESS = "claude_code_deepseek"

# Tasks that are non_agent_web by task_family but use CLI runner (no browser needed).
# These are pure text-in/code-out or text-in/text-out tasks evaluated by LLM judge.
CLI_NON_AGENT_TASKS = {34, 35, 36, 37, 38, 39}

def find_task_dirs():
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
        if num in SKIP_TASKS:
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

def run_one_episode(episode_id, task_dir, instance_path, runner_mode,
                    platform, model, harness, dry_run=False, no_docker=False):
    cmd = [
        sys.executable, str(RUNNER_DIR / "run_episode.py"), episode_id,
        "--runner", runner_mode,
        "--instance", str(instance_path),
        "--platform", platform,
        "--model", model,
    ]
    if runner_mode == "cli":
        cmd += ["--harness", harness, "--timeout", "1800"]
    if no_docker:
        cmd += ["--no-docker"]

    print(f"\n{'='*60}")
    print(f"  [{episode_id}] {task_dir.name}/{instance_path.name} "
          f"runner={runner_mode}")
    print(f"{'='*60}\n", flush=True)

    if dry_run:
        print("  [DRY RUN] skip", flush=True)
        return 0

    ep_dir = EPISODES_DIR / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT,
        stdout=open(ep_dir / "orchestrator_stdout.log", "w"),
        stderr=open(ep_dir / "orchestrator_stderr.log", "w"),
    )
    elapsed = time.time() - t0

    meta = {
        "episode_id": episode_id, "task": task_dir.name,
        "runner_mode": runner_mode, "instance": str(instance_path),
        "exit_code": result.returncode, "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(ep_dir / "batch_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"  [{episode_id}] {status} ({elapsed:.0f}s)", flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Batch run all FlowBenchmark episodes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--harness", default=DEFAULT_HARNESS,
                        help=f"Harness preset for CLI tasks (default: {DEFAULT_HARNESS})")
    parser.add_argument("--no-docker", action="store_true", default=False,
                        help="Skip Docker lifecycle for task 40 (Juice Shop)")
    args = parser.parse_args()

    tasks = find_task_dirs()
    if args.task:
        tasks = {args.task: tasks[args.task]} if args.task in tasks else {}

    total_ep = len(tasks) * SAMPLES_PER_TASK
    print(f"Tasks: {len(tasks)} | Episodes: {total_ep} "
          f"(est. {total_ep * 45 // 60} min)")
    print(f"Platform: {args.platform} | Model: {args.model}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    results = {}
    count = 0
    t0 = time.time()

    for task_num in sorted(tasks):
        task_dir = tasks[task_num]

        # Determine runner mode
        if task_num <= 14:
            runner = "web"
        elif 15 <= task_num <= 21 or task_num in CLI_NON_AGENT_TASKS:
            runner = "cli"
        elif 23 <= task_num <= 32:
            runner = "web"  # 023-032 sometimes also agent, but default web
        else:
            runner = "cli"  # 033+ tasks are agent/CLI by default

        instances = find_instance_files(task_dir)
        if len(instances) < SAMPLES_PER_TASK:
            print(f"  WARN: {task_dir.name} only {len(instances)} instances",
                  flush=True)

        task_rc = []
        for idx, inst in enumerate(instances):
            count += 1
            eid = f"{'W' if runner == 'web' else 'A'}{task_num:02d}{idx:02d}"

            clean_netns(eid)
            rc = run_one_episode(eid, task_dir, inst, runner,
                                args.platform, args.model, args.harness,
                                args.dry_run, args.no_docker)
            task_rc.append(rc)

            if not args.dry_run:
                clean_netns(eid)
                time.sleep(1)

        results[task_num] = task_rc

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  BATCH DONE: {count} episodes in {elapsed/60:.0f} min")
    print(f"{'='*60}\n")
    for num in sorted(results):
        rc_list = results[num]
        ok = sum(1 for r in rc_list if r == 0)
        print(f"  {tasks[num].name:40s} OK={ok}/{len(rc_list)}")

    report = {
        "episodes": count,
        "total_s": elapsed,
        "platform": args.platform,
        "model": args.model,
        "tasks": {
            str(n): {
                "dir": tasks[n].name,
                "ok": sum(1 for r in rc_list if r == 0),
                "results": rc_list,
            }
            for n, rc_list in results.items()
        },
    }
    with open(EPISODES_DIR / "batch_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: episodes/batch_report.json")

if __name__ == "__main__":
    main()
