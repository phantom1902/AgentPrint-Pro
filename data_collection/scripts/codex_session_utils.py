#!/usr/bin/env python3
"""Helpers for resolving and copying Codex session history files."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


SESSIONS_ROOT = Path.home() / ".codex" / "sessions"


@dataclass
class SessionRef:
    label: str
    session_id: str
    source_path: Path


def _read_session_meta(session_file: Path) -> dict | None:
    try:
        with session_file.open("r", encoding="utf-8") as handle:
            first = handle.readline().strip()
    except OSError:
        return None
    if not first:
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    if obj.get("type") != "session_meta":
        return None
    payload = obj.get("payload")
    return payload if isinstance(payload, dict) else None


def find_session_file_by_id(session_id: str, sessions_root: Path = SESSIONS_ROOT) -> Path | None:
    if not session_id or not sessions_root.exists():
        return None
    for session_file in sessions_root.rglob("*.jsonl"):
        payload = _read_session_meta(session_file)
        if not payload:
            continue
        if payload.get("id") == session_id:
            return session_file
    return None


def find_latest_session_for_workspace(
    workspace_dir: Path,
    started_at: float | None = None,
    sessions_root: Path = SESSIONS_ROOT,
) -> tuple[str, Path] | None:
    if not sessions_root.exists():
        return None
    workspace = str(workspace_dir.resolve())
    best: tuple[float, str, Path] | None = None
    for session_file in sessions_root.rglob("*.jsonl"):
        payload = _read_session_meta(session_file)
        if not payload or payload.get("cwd") != workspace:
            continue
        session_id = payload.get("id")
        if not isinstance(session_id, str):
            continue
        mtime = session_file.stat().st_mtime
        if started_at is not None and mtime + 30 < started_at:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, session_id, session_file)
    if best is None:
        return None
    return best[1], best[2]


def copy_episode_sessions(
    output_dir: Path,
    session_refs: list[SessionRef],
    clear_existing: bool = True,
) -> list[Path]:
    session_dir = output_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for old in session_dir.glob("*.jsonl"):
            old.unlink(missing_ok=True)
        for old in session_dir.glob("session_manifest.json"):
            old.unlink(missing_ok=True)

    copied: list[Path] = []
    manifest: list[dict] = []
    for idx, ref in enumerate(session_refs, start=1):
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in ref.label).strip("_") or f"session_{idx}"
        dst = session_dir / f"{idx:02d}_{safe_label}_{ref.session_id}.jsonl"
        shutil.copy2(ref.source_path, dst)
        copied.append(dst)
        manifest.append({
            "index": idx,
            "label": ref.label,
            "session_id": ref.session_id,
            "source_path": str(ref.source_path),
            "copied_path": str(dst),
        })

    (session_dir / "session_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return copied
