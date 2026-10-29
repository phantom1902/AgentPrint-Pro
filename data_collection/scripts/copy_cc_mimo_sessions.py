#!/usr/bin/env python3
"""
Copy Claude Code session JSONL files into each cc_mimo episode's session/ directory.

Logic:
- For each cc_mimo episode that has agent_live.log:
  - Find the corresponding .claude/projects/ workspace directory
  - List all session JSONL files
  - Filter out sessions whose FIRST assistant message is an API error
  - Among remaining valid sessions:
    * If only 1: use it
    * If multiple: pick the latest by timestamp
    * If the latest doesn't match agent_live.log content, try earlier sessions
      to find the one that DOES match (i.e., the session that produced the log)
  - Copy selected session to episode/session/ directory
  - Create session_manifest.json

Usage:
    python scripts/copy_cc_mimo_sessions.py [--dry-run] [--episode A1700]
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = PROJECT_ROOT / "episodes" / "Agent" / "cc_mimo"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Multi-turn tasks: agent engages in back-and-forth conversation within one session.
# For these tasks, multiple sessions might contain useful partial multi-turn progress,
# so we keep ALL valid (non-API-error) sessions.
# - Task 18: web_navigation_webarena (multi-step browser interaction)
# - Task 20: office_agent_multiwoz (multi-turn dialogue)
# - Task 21: general_assistant_gaia (multi-step assistant)
MULTI_TURN_TASKS = {18, 20, 21}


def extract_episode_task(episode_id):
    """Extract task number from episode ID like A1700 -> 17."""
    match = re.match(r'A(\d{2})\d{2}', episode_id)
    if match:
        return int(match.group(1))
    return None


def get_workspace_project_dir(episode_id):
    """Get the .claude/projects directory for this episode's workspace."""
    workspace = (EPISODES_DIR / episode_id / "workspace").resolve()
    if not CLAUDE_PROJECTS.is_dir():
        return None
    # Session metadata records the actual workspace independently of host names
    # and the CLI's directory-name encoding.
    for project in sorted(CLAUDE_PROJECTS.iterdir()):
        if not project.is_dir():
            continue
        for session in sorted(project.glob("*.jsonl")):
            try:
                with session.open() as stream:
                    for line in stream:
                        try:
                            entry = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        cwd = entry.get("cwd") if isinstance(entry, dict) else None
                        if isinstance(cwd, str) and Path(cwd).resolve() == workspace:
                            return project
            except (OSError, UnicodeError):
                continue
    return None


def is_api_error_session(session_path):
    """Check if a session's FIRST assistant message is an API error.

    Returns True only if the very first assistant message has isApiErrorMessage=true.
    Sessions whose first assistant message is a normal response (thinking/text)
    are NOT considered API error sessions, even if later messages contain errors.
    """
    try:
        with open(session_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "assistant":
                    return bool(entry.get("isApiErrorMessage", False))
    except Exception:
        pass
    return False


def get_session_timestamp(session_path):
    """Get the first timestamp from a session file."""
    try:
        with open(session_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp")
                if ts:
                    return ts
    except Exception:
        pass
    return "1970-01-01T00:00:00.000Z"


def get_first_assistant_text(session_path):
    """Get the first assistant message text/thinking from a session file."""
    try:
        with open(session_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "assistant" and "message" in entry:
                    msg = entry["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        for item in msg["content"]:
                            if isinstance(item, dict):
                                text = item.get("text", "")
                                thinking = item.get("thinking", "")
                                if thinking:
                                    return thinking
                                if text:
                                    return text
    except Exception:
        pass
    return ""


def read_agent_log_first_content(agent_log_path):
    """Extract the first THINKING or OUTPUT content from agent_live.log.

    Returns (content_type, content_text) or (None, None).
    """
    if not agent_log_path.exists():
        return None, None
    try:
        with open(agent_log_path) as f:
            first_line = f.readline().strip()
        match = re.match(r'\[\d{2}:\d{2}:\d{2}\.\d{3}\] (THINKING|OUTPUT): (.+)', first_line)
        if match:
            return match.group(1), match.group(2)
    except Exception:
        pass
    return None, None


def session_matches_agent_log(session_path, agent_log_path):
    """Check if session's first assistant content matches agent_live.log's first entry."""
    log_type, log_content = read_agent_log_first_content(agent_log_path)
    if not log_content or len(log_content) < 20:
        return False

    session_text = get_first_assistant_text(session_path)
    if not session_text:
        return False

    # Check if agent_live.log content prefix appears in session text, or vice versa
    # Use first ~60 chars for matching (enough to be unique but robust against minor differences)
    prefix_len = min(60, len(log_content))
    log_prefix = log_content[:prefix_len]

    return log_prefix in session_text


def get_first_user_message(session_path):
    """Get the first user message timestamp and content snippet for debugging."""
    try:
        with open(session_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "user" and "message" in entry:
                    ts = entry.get("timestamp", "")
                    content = entry["message"].get("content", "")
                    return ts, content[:80]
    except Exception:
        pass
    return "", ""


def find_best_sessions(session_files, episode_dir):
    """Find the best session(s) to keep for an episode.

    For multi-turn tasks: keep ALL valid (non-API-error) sessions.
    For non-multi-turn tasks: keep the single best session matching agent_live.log.

    Returns a list of selected session file paths (may be empty).
    """
    episode_id = episode_dir.name
    task_num = extract_episode_task(episode_id)
    agent_log = episode_dir / "agent_live.log"
    is_multi_turn = task_num in MULTI_TURN_TASKS

    if not session_files:
        return []

    # 1. Separate valid sessions from API-error sessions
    valid_sessions = []
    api_error_sessions = []
    for sf in session_files:
        if is_api_error_session(sf):
            api_error_sessions.append(sf)
        else:
            valid_sessions.append(sf)

    # 2. If no valid sessions, fall back to latest overall
    if not valid_sessions:
        session_files_sorted = sorted(session_files, key=get_session_timestamp)
        return [session_files_sorted[-1]]

    # 3. Sort valid sessions by timestamp (earliest first)
    valid_sorted = sorted(valid_sessions, key=get_session_timestamp)

    # 4. Multi-turn tasks: keep all valid sessions
    if is_multi_turn:
        return valid_sorted

    # 5. Non-multi-turn tasks: pick the best single session
    if len(valid_sorted) == 1:
        return valid_sorted

    # Multiple valid sessions — try to find the one matching agent_live.log
    latest = valid_sorted[-1]

    if agent_log.exists():
        # If latest matches, use it
        if session_matches_agent_log(latest, agent_log):
            return [latest]

        # Otherwise, search from newest to oldest for a match
        for sf in reversed(valid_sorted[:-1]):
            if session_matches_agent_log(sf, agent_log):
                print(f"    ℹ Latest session ({latest.name[:8]}...) doesn't match "
                      f"agent_live.log, using: {sf.name[:8]}...")
                return [sf]

        # No match found — still use latest
        print(f"    ⚠ No session matches agent_live.log; using latest: {latest.name[:8]}...")

    # Default: use the latest valid session
    return [latest]


def copy_session(episode_id, session_path, index):
    """Copy a session file to the episode's session/ directory."""
    episode_dir = EPISODES_DIR / episode_id
    session_dir = episode_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_id = session_path.stem
    label = "root_exec"

    dest_name = f"{index:02d}_{label}_{session_id}.jsonl"
    dest_path = session_dir / dest_name

    shutil.copy2(session_path, dest_path)

    return {
        "index": index,
        "label": label,
        "session_id": session_id,
        "source_path": str(session_path),
        "copied_path": str(dest_path),
    }


def process_episode(episode_id, dry_run=False):
    """Process one episode: find session and copy it."""
    episode_dir = EPISODES_DIR / episode_id
    if not episode_dir.exists():
        return None

    agent_log = episode_dir / "agent_live.log"
    if not agent_log.exists():
        return None  # Episode wasn't run

    proj_dir = get_workspace_project_dir(episode_id)
    if not proj_dir:
        print(f"  {episode_id}: No workspace project directory found")
        return None

    session_files = sorted(proj_dir.glob("*.jsonl"))
    if not session_files:
        print(f"  {episode_id}: No session files found")
        return None

    task_num = extract_episode_task(episode_id)
    n_sessions = len(session_files)
    is_multi_turn = task_num in MULTI_TURN_TASKS

    selected = find_best_sessions(session_files, episode_dir)

    if not selected:
        print(f"  {episode_id}: No valid session found")
        return None

    n_api = sum(1 for sf in session_files if is_api_error_session(sf))
    n_valid = n_sessions - n_api
    n_selected = len(selected)

    mt_label = " [multi-turn]" if is_multi_turn else ""
    extra = ""
    if n_sessions > 1:
        extra = f" (from {n_sessions} total: {n_valid} valid, {n_api} API-err)"

    print(f"  {episode_id} (task {task_num}){mt_label}: {n_sessions} session(s) → "
          f"{n_selected} selected{extra}")

    if dry_run:
        for sf in selected:
            is_err = is_api_error_session(sf)
            print(f"    → Would copy: {sf.name} (api_error={is_err})")
        return {"episode_id": episode_id, "copied": 0, "dry_run": True}

    manifest_entries = []
    for i, sf in enumerate(selected, 1):
        entry = copy_session(episode_id, sf, i)
        manifest_entries.append(entry)

    # Write session_manifest.json
    session_dir = EPISODES_DIR / episode_id / "session"
    manifest_path = session_dir / "session_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    return {
        "episode_id": episode_id,
        "copied": len(selected),
        "total_sessions": len(session_files),
    }


def main():
    parser = argparse.ArgumentParser(description="Copy cc_mimo session files to episodes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without copying")
    parser.add_argument("--episode", type=str,
                        help="Process a single episode (e.g., A1700)")
    args = parser.parse_args()

    episodes = []
    if args.episode:
        episodes = [args.episode]
    else:
        for d in sorted(EPISODES_DIR.iterdir()):
            if d.is_dir() and re.match(r'A\d{4}', d.name):
                episodes.append(d.name)

    total_copied = 0
    total_episodes = 0
    skipped = 0
    multi_turn_eps = []

    for ep_id in episodes:
        result = process_episode(ep_id, dry_run=args.dry_run)
        if result is None:
            skipped += 1
            continue
        total_episodes += 1
        if not args.dry_run:
            total_copied += result.get("copied", 0)
            if result.get("copied", 0) > 1:
                multi_turn_eps.append((ep_id, result["copied"], result["total_sessions"]))

    print()
    print(f"{'=' * 60}")
    action = "Would copy" if args.dry_run else "Copied"
    print(f"Processed: {total_episodes} episodes")
    if not args.dry_run:
        print(f"{action}: {total_copied} sessions")
    print(f"Skipped: {skipped} (no agent_live.log or sessions)")
    if multi_turn_eps:
        print(f"Multi-turn episodes (kept multiple sessions): {len(multi_turn_eps)}")
        for ep, n, total in multi_turn_eps[:10]:
            print(f"  {ep}: {n}/{total} sessions kept")
        if len(multi_turn_eps) > 10:
            print(f"  ... and {len(multi_turn_eps) - 10} more")


if __name__ == "__main__":
    main()
