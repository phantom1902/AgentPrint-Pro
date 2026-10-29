#!/usr/bin/env python3
"""Import task inputs from a tasks.zip or a tasks directory, preserve bundled code."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original tasks.zip or unpacked tasks/ directory")
    args = parser.parse_args()
    count = 0
    skipped = 0
    def write(relative, source):
        nonlocal count, skipped
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] != "tasks" or ".." in parts or "\\" in relative:
            raise ValueError(f"unsafe task path: {relative}")
        target = ROOT.joinpath(*parts)
        if not target.resolve().is_relative_to((ROOT / "tasks").resolve()):
            raise ValueError(f"path leaves tasks directory: {relative}")
        if target.exists():
            skipped += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as dest:
            shutil.copyfileobj(source, dest)
        count += 1
    if args.source.is_dir():
        for path in sorted(args.source.rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                with path.open("rb") as file:
                    write("tasks/" + path.relative_to(args.source).as_posix(), file)
    else:
        with zipfile.ZipFile(args.source) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as file:
                    write(info.filename, file)
    print(f"Imported {count} files, preserved {skipped} existing files")


if __name__ == "__main__":
    main()
