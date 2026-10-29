#!/usr/bin/env python3
"""Audit tracked files and optional Git history before an anonymous release.

Keep project-author identifiers in a private file outside this repository.
This check detects explicit identifiers, not every possible inference of identity.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
EMAIL = re.compile(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL = re.compile(r"https?://[^\s<>\"'`)]+")
LOCAL_PATH = re.compile(r"/(?:home|Users|mnt)/[^\s\"'`<>]+|[A-Za-z]:\\(?:Users|Documents and Settings)\\")
SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identifiers-file', type=Path,
                        help='Private newline-separated names, emails, handles, or machine paths')
    parser.add_argument('--check-history', action='store_true')
    args = parser.parse_args()
    blocked = []
    if args.identifiers_file:
        path = args.identifiers_file.resolve()
        if ROOT == path or ROOT in path.parents:
            parser.error('Store the identifier list outside the repository')
        blocked = [line.strip().casefold() for line in path.read_text().splitlines() if line.strip()]
    reviewed = json.loads((ROOT / 'configs/anonymity_allowlist.json').read_text())
    failures = []
    paths = [p for p in git('ls-files', '-z').split('\0') if p]
    for rel in paths:
        path = ROOT / rel
        if path.is_symlink():
            failures.append((rel, 'symlink requires review'))
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            failures.append((rel, 'binary metadata requires review'))
            continue
        except FileNotFoundError:
            failures.append((rel, 'tracked file missing'))
            continue
        if any(term in (rel + '\n' + content).casefold() for term in blocked):
            failures.append((rel, 'project-author identifier'))
        if LOCAL_PATH.search(content):
            failures.append((rel, 'machine-specific absolute path'))
        if SECRET.search(content):
            failures.append((rel, 'credential-like value'))
        if rel == 'configs/anonymity_allowlist.json':
            continue
        for kind, regex in [('emails', EMAIL), ('urls', URL)]:
            allowed = set(reviewed[kind].get(rel, []))
            if set(regex.findall(content)) - allowed:
                failures.append((rel, 'unreviewed ' + kind))
    if args.check_history:
        commits = git('rev-list', '--all').splitlines()
        for sha in commits:
            metadata = git('show', '-s', '--format=%an%n%ae%n%cn%n%ce%n%B', sha)
            identity = metadata.splitlines()[:4]
            if identity != ['Anonymous Authors', 'anonymous@example.invalid'] * 2:
                failures.append(('Git history', 'non-anonymous author or committer'))
            if any(term in metadata.casefold() for term in blocked):
                failures.append(('Git history', 'project-author identifier in metadata'))
            # This release uses one root snapshot so deleted historical content
            # cannot reintroduce the original identities through reachable parents.
            if len(git('rev-list', '--parents', '-n', '1', sha).split()) != 1:
                failures.append(('Git history', 'parent history needs a separate content audit'))
        if len(commits) != 1:
            failures.append(('Git history', 'expected one anonymous release snapshot'))
    for path, reason in failures:
        print('FAIL:', path, '-', reason)
    if failures:
        raise SystemExit(1)
    print(f'PASS: {len(paths)} tracked text files; reviewed links and third-party contacts.')
    if args.check_history:
        print('PASS: one anonymous root commit across all local refs.')
    print('Scope: explicit identifiers only; hosting-account identity and cached views are separate.')


if __name__ == '__main__':
    main()
