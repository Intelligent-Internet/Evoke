# SAE Research Archive

This is the single repository entrypoint for II-42 SAE model, representation,
retrieval, and index-architecture research. It keeps conclusions, evidence
metadata, and reproducibility metadata together without making historical
experiments part of the product runtime.

The first-release prerelease model reports now live with the product
documentation: [English](../technical-report-evoke-model.md) and
[Traditional Chinese](../technical-report-evoke-model-zh.md). Their frozen P2.1
evaluation sources remain here; actionable follow-up lives in
[Model Planning](../model-planning.md) under the product roadmap.

## Structure

- [Reports](reports/README.md) contains plans, contracts, experiment reports,
  architecture designs, milestone summaries, and route-closure decisions.
  Numbered work is grouped in `Mxxxx-Mxxxx` ranges; cross-cutting architecture
  and milestone documents use the `designs/` and `milestones/` collections.
- [Artifacts](artifacts/README.md) contains the integrity manifest for external
  JSON, logs, matrices, checkpoints, and other generated model evidence. Large
  artifacts remain outside Git and are restored by manifest identity.
- [Source](source/README.md) contains the integrity manifest for frozen
  historical trainers, runners, probes, and their dependency closure. Product
  code must not import these retired experiment sources.

## Authority

This tree is historical research evidence, not a product API or active product
TODO queue. The current storage and lifecycle authority is the
[Convergent Segmented Index](../convergent-segmented-index.md).

New SAE research should add a report or contract under `reports/`, write live
outputs to the ignored `runs/` directory, and archive durable machine evidence
through `artifacts/manifest.jsonl`. Do not add another `docs/research-*` root.
