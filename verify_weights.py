#!/usr/bin/env python3
"""Compare supplied checkpoints against the local experimental starting files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=root / 'weights')
    args = p.parse_args()
    ok = True
    for entry in json.loads((root / 'WEIGHTS.json').read_text()):
        path = args.weights / entry['suite_filename']
        if not path.is_file():
            print('MISSING', path.name)
            ok = False
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        matches = h.hexdigest() == entry['sha256']
        print('MATCH' if matches else 'DIFFERENT', path.name, h.hexdigest())
        ok = ok and matches
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
