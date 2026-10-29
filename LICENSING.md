# Licensing scope

The repository contains separately licensed code, data, and third-party implementations. A root license does not replace the rights or notices attached to third-party material.

## Project-owned code and documentation

Copyright (c) 2026 Anonymous Authors.

The benchmark authors' original code and documentation are licensed under the GNU General Public License, **version 3 only** (`GPL-3.0-only`); see [LICENSE](LICENSE). This grant covers the authors' original contributions to the experiment controller, adapters, evaluation utilities, configurations, tests, documentation, and the collection code in `data_collection/`. For adapted files, the grant covers only contributions the authors own; upstream portions remain subject to their own terms.

You may use, modify, and redistribute this code under GPL-3.0-only. Distribution of covered modified or combined works must comply with the GPL, including applicable source-code and notice requirements. This is not a claim that all bundled third-party implementations have been cleared for redistribution. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), particularly the unresolved FlowLens, YaTC, and NetMamba entries.

## ANT dataset

The authors' rights in the separately released ANT dataset and original annotations, including the fixed `Benchmark/data/ANT/{train,val,test}` splits, are licensed under **Creative Commons Attribution 4.0 International** (`CC-BY-4.0`). See [LICENSE-DATA](LICENSE-DATA) for the full text and the [official license summary](https://creativecommons.org/licenses/by/4.0/).

For the anonymous release, the designated attribution is **“ANT dataset — Anonymous Authors.”** Retain the supplied license and attribution notices, link to the version of the data used when a release link is supplied, and indicate modifications. No identifying author contact or account URL is required in this anonymous artifact. Include this notice and `LICENSE-DATA` with the separate data distribution.

This grant covers only rights held by the benchmark authors. It does not relicense third-party task datasets, website content, software, model weights, or third-party content present in captured payloads. Those materials retain their own terms and any other applicable rights. The collection task index identifies task inputs; listing an input is not a license grant for it.

## Scope boundaries

- Code and documentation: GPL-3.0-only for original project contributions, with third-party exceptions described above.
- Dataset and original annotations: CC-BY-4.0 to the extent of the authors' rights; this code repository does not bundle the ANT dataset.
- Upstream baseline code and dependencies: their respective licenses and notices; unknown permission remains unresolved.
- Model checkpoints: not included and not licensed by either root grant.

Both licenses permit commercial use under their respective terms. There is no additional “research only” or “non-commercial” restriction. Licensing does not change the repository's hosting visibility.
