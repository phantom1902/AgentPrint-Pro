#!/usr/bin/env python3
"""Check the exported runtime without creating namespaces or calling an agent."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="skip installed module/binary checks")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    errors = []
    sources = list((ROOT / "scripts").rglob("*.py"))
    local = {p.stem for p in sources}
    available = {p.name for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts}
    external = set()
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append("missing file: " + item["path"])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append("changed packaged file: " + item["path"])
    for path in sources:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                external.add(node.module.split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Literal script references, including subprocess entry points.
                value = node.value
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*\.(py|sh)", value):
                    if Path(value).name not in available:
                        errors.append(f"unresolved script reference: {path.name}: {value}")
    for name in sorted(external - local - sys.stdlib_module_names):
        if not args.source_only and importlib.util.find_spec(name) is None:
            errors.append("missing Python module: " + name)
    for path in (ROOT / "scripts").rglob("*.sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr.strip())
        if not path.stat().st_mode & 0o111:
            errors.append("script is not executable: " + str(path))
    mocks = list((ROOT / "scripts" / "mock").glob("mock_*_server.py"))
    if len(mocks) != 8:
        errors.append(f"expected 8 mock services, found {len(mocks)}")
    if not args.source_only:
        for executable in ("sudo", "ip", "iptables", "tcpdump", "bwrap", "git", "curl", "jq"):
            if not shutil.which(executable):
                errors.append("missing executable: " + executable)
    print(json.dumps({"files": len(manifest["files"]), "mock_services": len(mocks),
                      "third_party_modules": sorted(external - local - sys.stdlib_module_names),
                      "errors": errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
