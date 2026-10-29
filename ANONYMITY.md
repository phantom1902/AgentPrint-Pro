# Anonymous release audit

This release has been checked for explicit identifiers of the benchmark authors in tracked filenames, text files, configuration records, links, and reachable Git commit metadata.

## Changes

- Replaced development-machine paths in TrafficFormer helper code with paths relative to the working directory or the requested output file.
- Replaced internal experiment directory names and run timestamps in configuration provenance with descriptive source labels. Evidence hashes and the reproducibility qualifications remain available.
- Removed the local validation date from the release record.
- Published the sanitized files as one root commit with an anonymous author and committer. Original history is not a parent of that snapshot.
- Retained third-party copyright notices, licenses, citations, and upstream contact information. Third-party documentation is explicitly labeled as such; it does not identify the benchmark authors.

## Collection import

The collection import excludes macOS resource metadata, a private development note, an invalid nested archive, and empty source files. Session discovery no longer embeds the original development-machine path. Provider URLs and local mock URLs are reviewed references, not author contacts; credentials remain environment lookups. Original project license attribution uses “Anonymous Authors,” while third-party notices remain intact. Generated sessions and browser/account state are excluded from the source package.

## Recheck

Keep a newline-separated list of author names, email addresses, account names, and machine identifiers **outside this repository**. Then run:

```bash
python audit_anonymity.py --identifiers-file /path/to/private-identifiers.txt --check-history
```

The checker scans tracked files for the supplied identifiers, machine paths, credential-like values, and unreviewed emails or URLs. Reviewed third-party references are recorded by file in `configs/anonymity_allowlist.json`. The history check verifies that all local refs resolve to a single root commit with anonymous author and committer fields. Review any later commits before producing another anonymous snapshot.

The checker does not infer unknown coauthors or establish that a project cannot be identified from its scientific content. An empty identifier list is not equivalent to checking all authors.

## Dataset release account

The README links to [the ANT dataset release](https://huggingface.co/datasets/benzenenene/ANT). The repository owner states that this account was created specifically for the submission, is not linked to author identities, and has no unrelated prior contribution history. Its publicly visible profile was checked and displays the name “anonymous,” no organizations, and no public repositories or discussions at the time of inspection.

Unauthenticated requests for the dataset metadata, file tree, dataset card, and commit history returned HTTP 401 during this audit. The account statement is supplied by the owner; this repository audit does not establish the anonymity of the hosted dataset, its packet contents, or its unavailable history. Dataset access for reviewers must be checked separately.

## Repository-wide recheck

- Scanned all 250 tracked files, including filenames, hidden files, source comments, vendored documentation, package metadata, provenance manifests, and license notices. All tracked files are regular UTF-8 text; no tracked symlinks, binary attachments, embedded archives, session dumps, or credential files were found.
- Checked known author identifiers and machine paths, including URL-encoded, HTML-escaped, Unicode-escaped, and normalized forms. Reviewed emails, URLs, and credential-like strings; the dataset URL is an explicitly approved release link, and retained personal references belong to cited third-party projects.
- Checked remote branches, tags, contributor metadata, releases, issues/pull requests, commit comments, workflow runs, deployments, and Pages. The repository has one anonymous root snapshot on `main`; the other inspected publication surfaces have no content. No wiki repository was accessible.
- Checked a downloaded source archive against the tracked file hashes. These checks cover the current published source snapshot, not the hosting account's identity or historical objects retained by the hosting service.

## Submission boundary

Share the anonymized service URL or a sanitized source archive with reviewers. A GitHub repository URL identifies its owner even when commit authors are anonymous. Do not include `.git`, private backups, raw experiment outputs, or the private identifier list in the review artifact. Newly generated logs can contain paths from the machine running the experiments and need their own review before publication.

Rewriting a branch does not erase old copies or cached commit views held by a hosting provider. This audit covers the sanitized file tree and the new reachable snapshot, not those copies. The final anonymous service page and its downloadable archive must also be checked after the service has imported the new snapshot.
