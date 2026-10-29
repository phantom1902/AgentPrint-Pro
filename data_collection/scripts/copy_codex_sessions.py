#!/usr/bin/env python3
"""Backfill raw Codex session JSONL files into episode/session/ directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_session_utils import SessionRef, copy_episode_sessions, find_latest_session_for_workspace, find_session_file_by_id


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "episodes" / "codex"


def _safe_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_session_ids_from_jsonl(path: Path) -> list[str]:
    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "thread.started" and isinstance(obj.get("thread_id"), str):
                    ids.append(obj["thread_id"])
                elif obj.get("type") == "session_meta":
                    payload = obj.get("payload") or {}
                    if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                        ids.append(payload["id"])
    except OSError:
        return []
    return ids


def _extract_child_session_ids(path: Path) -> list[str]:
    ids: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                item = obj.get("item")
                if not isinstance(item, dict):
                    continue
                receivers = item.get("receiver_thread_ids")
                if isinstance(receivers, list):
                    for rid in receivers:
                        if isinstance(rid, str):
                            ids.append(rid)
    except OSError:
        return []
    return ids


def _find_local_session_file_by_id(episode_dir: Path, session_id: str) -> Path | None:
    if not session_id:
        return None
    for session_file in episode_dir.rglob("*.jsonl"):
        if "session" not in session_file.parts and ".sandbox_home" not in session_file.parts:
            continue
        try:
            first = session_file.read_text(encoding="utf-8").splitlines()[0]
            obj = json.loads(first)
        except Exception:
            continue
        payload = obj.get("payload") or {}
        if obj.get("type") == "session_meta" and isinstance(payload, dict) and payload.get("id") == session_id:
            return session_file
    return None


def _find_sessions_by_episode_id(episode_id: str) -> list[tuple[float, str, Path]]:
    results: list[tuple[float, str, Path]] = []
    suffix = f"/{episode_id}/workspace"
    for session_file in (Path.home() / ".codex" / "sessions").rglob("*.jsonl"):
        try:
            first = session_file.read_text(encoding="utf-8").splitlines()[0]
            obj = json.loads(first)
        except Exception:
            continue
        payload = obj.get("payload") or {}
        cwd = payload.get("cwd")
        sid = payload.get("id")
        if not (isinstance(cwd, str) and isinstance(sid, str)):
            continue
        if cwd.endswith(suffix) or f"{suffix}/" in cwd:
            results.append((session_file.stat().st_mtime, sid, session_file))
    return results


def gather_session_refs(episode_dir: Path) -> list[SessionRef]:
    refs: list[SessionRef] = []
    seen_ids: set[str] = set()

    def add(label: str, session_id: str | None) -> None:
        if not session_id or session_id in seen_ids:
            return
        session_file = find_session_file_by_id(session_id)
        if session_file is None:
            session_file = _find_local_session_file_by_id(episode_dir, session_id)
        if session_file is None:
            return
        seen_ids.add(session_id)
        refs.append(SessionRef(label=label, session_id=session_id, source_path=session_file))

    run_meta = _safe_json(episode_dir / "run_meta.json")
    if isinstance(run_meta.get("session_id"), str):
        add("run_meta", run_meta["session_id"])

    cli_response = _safe_json(episode_dir / "cli_response.json")
    turns = cli_response.get("turns")
    if isinstance(turns, list):
        for idx, turn in enumerate(turns, start=1):
            if isinstance(turn, dict) and isinstance(turn.get("session_id"), str):
                add(f"turn_{idx:02d}", turn["session_id"])

    root_jsonl = episode_dir / "codex_exec.jsonl"
    if root_jsonl.exists():
        for session_id in _extract_session_ids_from_jsonl(root_jsonl):
            add("root_exec", session_id)
        for session_id in _extract_child_session_ids(root_jsonl):
            add("root_child", session_id)

    for jsonl in sorted(episode_dir.rglob("*.codex_exec.jsonl")):
        label = jsonl.name.replace(".codex_exec.jsonl", "")
        for session_id in _extract_session_ids_from_jsonl(jsonl):
            add(label, session_id)
        for session_id in _extract_child_session_ids(jsonl):
            add(f"{label}_child", session_id)

    if not refs:
        workspace = episode_dir / "workspace"
        if workspace.exists():
            latest = find_latest_session_for_workspace(workspace, None)
            if latest is not None:
                add("workspace_latest", latest[0])

    if not refs:
        matches = sorted(_find_sessions_by_episode_id(episode_dir.name))
        if matches:
            _, session_id, session_file = matches[-1]
            seen_ids.add(session_id)
            refs.append(SessionRef(label="episode_id_latest", session_id=session_id, source_path=session_file))
            for child_id in _extract_child_session_ids(session_file):
                add("episode_id_latest_child", child_id)

    if not refs:
        local_candidates: list[tuple[float, str, Path]] = []
        for session_file in episode_dir.rglob("*.jsonl"):
            if ".sandbox_home" not in session_file.parts:
                continue
            try:
                first = session_file.read_text(encoding="utf-8").splitlines()[0]
                obj = json.loads(first)
            except Exception:
                continue
            payload = obj.get("payload") or {}
            sid = payload.get("id")
            if obj.get("type") == "session_meta" and isinstance(payload, dict) and isinstance(sid, str):
                local_candidates.append((session_file.stat().st_mtime, sid, session_file))
        local_candidates.sort()
        if local_candidates:
            _, session_id, session_file = local_candidates[-1]
            seen_ids.add(session_id)
            refs.append(SessionRef(label="local_latest", session_id=session_id, source_path=session_file))
            for child_id in _extract_child_session_ids(session_file):
                add("local_latest_child", child_id)

    return refs


def resolve_episode_dirs(episodes_root: Path, episodes: list[str]) -> list[Path]:
    if episodes:
        result = []
        for item in episodes:
            path = Path(item)
            result.append(path.resolve() if path.is_dir() else (episodes_root / item).resolve())
        return result
    return sorted(path for path in episodes_root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy raw Codex sessions into episode/session/")
    parser.add_argument("episodes", nargs="*", help="Episode ids or paths under episodes/codex")
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    args = parser.parse_args()

    episodes_root = Path(args.episodes_root).resolve()
    if not episodes_root.exists():
        raise SystemExit(f"Episodes root not found: {episodes_root}")

    episode_dirs = resolve_episode_dirs(episodes_root, args.episodes)
    ok = 0
    miss = 0
    for episode_dir in episode_dirs:
        refs = gather_session_refs(episode_dir)
        if not refs:
            print(f"[MISS] {episode_dir.name}: no matching Codex sessions found")
            miss += 1
            continue
        copied = copy_episode_sessions(episode_dir, refs)
        print(f"[OK] {episode_dir.name}: copied {len(copied)} session file(s)")
        ok += 1
    print(f"OK={ok} MISS={miss}")
    return 0 if ok or not episode_dirs else 1


if __name__ == "__main__":
    raise SystemExit(main())
