# Historical SAE Research Source Archive

Updated: 2026-10-10

## New Repository Snapshots

The [October 10 workspace archive](../reports/milestones/research-workspace-archive-2026-10-10.md)
adds small research sources and conclusions to Git without changing the older
external archives below. Its [checksum inventory](snapshots/2026-10-10-workspace.json)
records original and retained paths for all 105 previously untracked files.

- [M217-M233 source notes](m0200-m0299/README.md) describe the retained
  trainers/launchers, incomplete external dependency closure, and replay limits.
- [Standalone prototypes](prototypes/README.md) preserve illustrative code,
  not validated product implementations.
- NG80-NG88 runners remain under `scripts/`, with matching `test_ng*` and
  `test_review_ng*` files under `tests/`. Existing imports and frozen-run
  source-copy conventions rely on these paths. CI excludes those research
  tests from the product unit-test selection; they are not product entrypoints.

Archiving does not authorize running a launcher, installing its dependencies,
accessing locked test data, resuming an interrupted run, or deploying a model.
New work must use a separately reviewed run, not mutate an archived attempt.

## Original External Archives

The public source tree contains supported product, release, benchmark, and
validation tooling. Historical M-series experiments remain reproducible
evidence, but their 1,000-plus runners, trainers, oracle probes, and one-off
patches are not product entrypoints.

The historical source was frozen before removal from Git. The archive has two
cleanup batches:

- initial source commit:
  `273832fecad72ce4cce8e9cf56fd66f237086692`;
- initial file count: `1,591`;
- initial bytes: `20,472,558`;
- initial checksum-list SHA-256:
  `571e503287dde506eefbc27a580ac3d605f0511c9fa05842a7edebdbd26387cf`.
- final residue source commit:
  `f9476560c53a0f4bc710e07fdc7602ea8cc1eb1f`;
- final residue file count: `4`;
- final residue bytes: `59,696`;
- final residue checksum-list SHA-256:
  `76b924a804f51123ce717182b36e926e2c1908f3250ee3edce05ee5becafdb7d`.

The combined manifest contains `1,595` files and `20,532,254` bytes. Its
SHA-256 is
`70e0904f86b1cf15c753857e751f25b8df14151d5439a793036b580639c921cd`
and is enforced by the product-convergence inventory.

The machine-readable [manifest](manifest.jsonl) maps each original path to its
source commit, archive key, byte size, media type, and SHA-256 digest. Formal
conclusions remain under
[`docs/research-sae/reports`](../reports/README.md), while generated
evidence uses the separate
[research artifact archive](../artifacts/README.md).

Maintainers store the exact source archive at:

```text
/Volumes/Betty/II42-research-source/
  273832fecad72ce4cce8e9cf56fd66f237086692/
  f9476560c53a0f4bc710e07fdc7602ea8cc1eb1f/
```

That path is operational storage, not a build, test, or release dependency.
To restore a historical experiment, select the original paths from the
manifest, copy their `archive_key` entries into a separate research checkout,
and verify every digest. Do not restore M-series scripts to the product tree.

The original external archive includes the source dependency closure and its matching research
tests, so retained production scripts and tests do not import or invoke removed
research sources. Current CI enforces that boundary.
