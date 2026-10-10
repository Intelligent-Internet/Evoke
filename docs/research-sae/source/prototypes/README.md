# Standalone Research Prototypes

`poc_block_max_wand.py` is a retained synthetic high-DF block-pruning simulation,
not the PostgreSQL index implementation or a release benchmark. No product
module imports it. It was moved here from `scripts/` without changing its bytes.
This also preserves 16 pre-existing trailing-whitespace lines reported by
`git diff --check`; no global whitespace or CI exemption was added.

The printed claim that document reordering is "strictly required" is an
unvalidated historical assertion, not a conclusion established by this archive.
The simulation renumbers documents after sorting, lacks an independent exact
top-k oracle, and does not count cursor movement through skipped postings as
`touched_postings`. Its counter and timing therefore do not establish native
query cost, result identity, or a general necessity theorem.

Retain it for mechanism discussion. Any further use requires a separate
controlled experiment; this archival pass did not execute it or modify the
product's Block-Max WAND path.
