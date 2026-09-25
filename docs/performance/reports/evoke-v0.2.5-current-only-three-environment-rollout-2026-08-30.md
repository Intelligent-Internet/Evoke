# Evoke v0.2.5 Current-Only Three-Environment Rollout

Date: 2026-08-30

Last updated: 2026-09-03

Status: **qualified** on Shadow, Elm, and Mac.

## Scope

This report closes the current-only fp16/U8/A0.50 rollout on the declared
development targets:

- Shadow PostgreSQL 18.4 primary and physical asynchronous standby, database
  `ii_dev`
- Elm PostgreSQL 18.6, database `ii_dev`
- Mac Homebrew PostgreSQL 18.6, database `postgres`

All planned roots are valid, ready, queryable, and use the current
`convergent_segments` format. The deployment-specific keys, INCLUDE columns,
predicates, `field_aware` choices, and preload priorities were preserved.

## Release Provenance

| Environment | Source | Package SHA-256 | Installed library SHA-256 | ORT contract |
| --- | --- | --- | --- | --- |
| Shadow | `e836e0d81d0e66476e8ec305a057f2096097dfff` | `65f83481e9915506d941581d8323fe905b7f9926cab1df1b23e4c0c0fa2ff6be` | `ba789b8e57a584c7e79a8399547d7e327176112efd4590a836c4c2b538211062` on primary and standby | API 29, linked and available 1.29.0; library SHA `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca` |
| Elm | `7d5a1e8f6d4d1ceb9cccf6d6ff0ac1b7b09ac019` | `7b84ee2047f2b50054c49dc59e4e60492a226b0ae0cf1c70298d4dc2b2f99783` | `07fa7e59a89348992e2a4ec46012008ab36b5e455171d69d2e80ec05183cf412` | API 29, linked and available 1.29.0; library SHA `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca` |
| Mac | `7d5a1e8f6d4d1ceb9cccf6d6ff0ac1b7b09ac019` | `6bce09c589a10e636faf6a9dc97756fb241c5056f32231019d8054a669211843` | `365be917f847b86d1da6c95ae9463d0ae4382415d5a0bc1b8b56c67ac90245c7` | API 29, linked and available 1.29.0; library SHA `70568a2fdf7be410976244cc113276d0345c5e04835154fc7e447be5ecacc46b` |

Package paths:

- Shadow:
  `/data/ii42-builds/ii42-release-e836e0d8-v1/src/dist/ii42-v0.2.5-linux-x86_64-pg18.zip`
- Elm:
  `/home/leask/ii42-builds/ii42-release-7d5a1e8f-v1/dist/ii42-v0.2.5-linux-x86_64-pg18.zip`
- Mac: `dist/ii42-v0.2.5-darwin-arm64-pg18.zip`

All three targets report extension and catalog version 0.2.5. Installed
libraries match their qualified package artifacts. No package-contract or model
signature mismatch remains.

## Root Inventory

### Shadow: 9/9

- `commons.data_arxiv__title_abstract__field_aware_bm25_idx`
- `commons.data_policy_ca__title_description__field_aware_bm25_idx`
- `commons.data_policy_ca_chunks__content__bm25_idx`
- `commons.data_policy_tx__title_description__field_aware_bm25_idx`
- `commons.data_policy_tx_chunks__content__bm25_idx`
- `commons.data_policy_wa__title_description__field_aware_bm25_idx`
- `commons.data_policy_wa_chunks__content__bm25_idx`
- `commons.data_pubmed__title_abstract__field_aware_bm25_idx`
- `commons.sys_chunks__content__bm25_idx`

The nine Commons product roots are valid, ready, current-only, and have exact
fp16/U8/A0.50 reloptions. The five obsolete benchmark/calibration roots were
removed separately.

### Elm: 10/10

- `commons.data_arxiv__title_abstract__field_aware_bm25_idx`
- `commons.data_policy_ca__title_description__field_aware_bm25_idx`
- `commons.data_policy_ca_chunks__content__bm25_idx`
- `commons.data_policy_tx__title_description__field_aware_bm25_idx`
- `commons.data_policy_tx_chunks__content__bm25_idx`
- `commons.data_policy_wa__title_description__field_aware_bm25_idx`
- `commons.data_policy_wa_chunks__content__bm25_idx`
- `commons.data_pubmed__title_abstract__field_aware_bm25_idx`
- `commons.sys_chunks__content__bm25_idx`
- `ii42_cf_eval.pubmed_250k_ii42_idx`

The PubMed accelerator publication finished successfully. All ten roots have
active postings, matching contracts and signatures, healthy lightweight
payload checks, and ready/current generation state.

### Mac: 19/19

- `ii42_beir15.docs_arguana_bm25_idx`
- `ii42_beir15.docs_fiqa_bm25_idx`
- `ii42_beir15.docs_nfcorpus_bm25_idx`
- `ii42_beir15.docs_scifact_bm25_idx`
- `ii42_broad10.docs_arguana_bm25_idx`
- `ii42_shared15.docs_arguana_bm25_idx`
- `ii42_shared15.docs_climate_fever_bm25_idx`
- `ii42_shared15.docs_cqadupstack_bm25_idx`
- `ii42_shared15.docs_dbpedia_entity_bm25_idx`
- `ii42_shared15.docs_fever_bm25_idx`
- `ii42_shared15.docs_fiqa_bm25_idx`
- `ii42_shared15.docs_hotpotqa_bm25_idx`
- `ii42_shared15.docs_msmarco_bm25_idx`
- `ii42_shared15.docs_nfcorpus_bm25_idx`
- `ii42_shared15.docs_nq_bm25_idx`
- `ii42_shared15.docs_quora_bm25_idx`
- `ii42_shared15.docs_scidocs_bm25_idx`
- `ii42_shared15.docs_scifact_bm25_idx`
- `ii42_shared15.docs_webis_touche2020_bm25_idx`

All nineteen roots have active postings, matching contracts and signatures,
healthy lightweight payload checks, and ready/current generation state.

No invalid or unready planned roots, unsupported roots, orphan candidate
indexes, or qualification canary schemas remain on the declared targets.

## Runtime Configuration

| Environment | Runtime workers | Accelerators | Maintenance limit | Final runtime state |
| --- | ---: | --- | ---: | --- |
| Shadow | 16 | `runtime_accelerators=[]`; local CPU | 4 | 16 ready, queue 0, failures 0, busy rejections 0 |
| Elm | 8 | 20 Azure services | 4 | 8 ready, queue 0, failures 0 |
| Mac | 2 | local CoreML; `runtime_accelerators=[]` | 4 | 2 ready, queue 0, failures 0 |

Elm retained 24 cumulative transient connection failures from the rollout, but
ended with zero consecutive failures, zero liveness timeouts, zero recoveries,
and no failed work. The Azure runtime fleet remains allocated pending the
user's retain-or-delete decision.

## Qualification Evidence

### Shadow

- Commit `e836e0d8` was packaged from an exact source archive and deployed
  standby first, then primary. PostgreSQL was stopped before each
  shared-preload library replacement. Rollback copies are retained under
  `/data/ii42-builds/ii42-release-e836e0d8-v1/rollback-predeploy` on both
  nodes. Both nodes now
  report the package hashes above, extension 0.2.5,
  `enabled:api=29`, and `available:linked:version=1.29.0`. Existing indexes
  were upgraded in place; no root was rebuilt.
- Pre-deployment validation passed: 490 Python tests with one declared skip,
  CTest 1/1, product-convergence inventory, convergent SAE lifecycle 80/80,
  transactional delta lifecycle 22/22, unified lifecycle 77/77, physical
  replication, shared-preload lifecycle closure, and `git diff --check`. The
  50,000-document medium lifecycle benchmark retained bounded build RSS and
  showed no material BM25 or SAE regression.
- The nine Commons roots remain valid, ready, `convergent_segments`, and
  payload healthy on primary and standby. All nine preserve exact
  fp16/U8/A0.50 reloptions, keys, INCLUDE columns, predicates, field-aware
  choices, and preload priorities. All report `query_metadata_warm=true` and
  accelerator `refresh_due=false`; there are no invalid, old-format,
  candidate, or temporary Evoke indexes.
- Before this deployment, the warm PubMed production-shaped query averaged
  approximately 5.76 seconds and foreground-merged 33,957 posting blocks,
  245,585 postings, and 51,462 documents from a stale baseline. The new path
  serves the compatible baseline once, revalidates current TID, predicate, and
  COW visibility, and performs no foreground posting merge. Trace now reports
  `accelerator_stale_baseline=true`, `posting_block_reads=0`, and
  `postings_examined=0`.
- The first post-restart arXiv read was cold at 2.91 seconds; subsequent runs
  were 169-180 ms. PubMed warm runs were initially 523-565 ms. During the full
  PubMed accelerator refresh, approximately 90 foreground samples stayed in
  the 0.59-0.66 second range. After publication, ten samples were 0.58-0.60
  seconds. A later final review under live Commons activity measured nine warm
  primary PubMed samples at 0.825-0.838 seconds and four warm standby samples
  at 0.820-0.825 seconds; the route and zero-posting trace were unchanged.
- The new scheduler selected PubMed accelerator work while semantic completion
  remained continuously actionable. A single, non-restarting build ran for
  approximately 27.5 minutes. At publication, the accelerator source advanced
  from manifest 70352 to 72983 and baseline sequence from 8,367,954 to
  8,504,228 while generation advanced from 72983 to 72984. Accelerator
  `refresh_due` changed from true to false even though live semantic debt kept
  growing. Subsequent semantic work advanced generation to 72987 while the
  compatible 72983 accelerator remained authoritative and fast.
- Broad planner filters now use the accelerator's authenticated scope postings
  before the exact PostgreSQL predicate recheck. The production-shaped PubMed
  2024 `Article` query avoided materializing 223,590 allowed TIDs: one probe
  examined 200 candidates, retained 200 matches, returned 50 rows, and left
  the old bitmap child unexecuted. Ten warm primary samples under concurrently
  observed maintenance were 817-825 ms; three warm standby samples were
  812-817 ms. The old exact-TID oracle took 29.84 seconds, but its top 50 and
  scores matched the planner route exactly. Unsupported predicates and
  incomplete probes still fall back to the original exact subset scorer.
- The accelerator build peaked near 21.5 GiB RSS and used about 101 GiB of
  temporary BufFile writes, but foreground latency stayed bounded. Temporary
  files were reclaimed on completion and `pgsql_tmp` is now zero bytes.
  Primary and standby have zero-byte WAL lag, 16/16 runtime workers are ready,
  runtime failures, recoveries, and busy rejections are zero, and external
  runtime accelerators remain disabled. Maintenance limit is restored to 4,
  timer interval to one minute, and periodic low-debt interval to one hour.
  Post-deployment warning logs contain no crash, corruption, replication, or
  Evoke runtime error.

### Elm

- Filtered arXiv and PubMed Commons queries returned correct rows in 23.75 and
  30.55 seconds respectively; an explicit arXiv query completed in 385 ms.
- CRUD canary insert, update, and delete visibility passed. Semantic pending
  work converged from 2 to 0 through background and supported explicit
  maintenance; the final gamma score was 8.066795 with zero failures.
- A normal lock arbitration `lock_busy` event resolved without intervention.
- PostgreSQL restarted successfully after an idle check. Shared-preload recovery
  completed in approximately two minutes, all ten roots remained ready/current,
  and a post-restart `sys_chunks` query returned three hits within its bounded
  timeout.

### Mac

- Filtered SciFact query smoke completed in 350 ms; an explicit query completed
  in 112 ms.
- CRUD canary insert, update, and delete visibility passed. Semantic pending
  work converged from 2 to 0 through supported maintenance, with the final gamma
  score increasing from 0.578 to 8.066791 and zero failures.
- Homebrew PostgreSQL restarted successfully and was ready in three seconds.
  All nineteen roots survived ready/current and post-restart filtered query
  smoke passed.

## Resource State

- Shadow has approximately 91 GiB available memory on both the primary and
  standby. The primary has 720 MiB swap in use and the standby 3.0 GiB,
  with no observed paging pressure or runaway RSS. `/data` has approximately
  2.9 TiB free on the primary and 3.3 TiB on the standby. Root filesystem use is
  11% and 5% respectively; build and deployment artifacts remain on `/data`.
- Elm has 201 GiB available memory after restart and 659 MiB swap in use.
  `/Volumes/Calf` is 83% used with 647 GiB free. The preload restart briefly
  peaked at 265.8 GiB memory and 28.5 GiB swap, then settled normally.
- Mac has approximately 69-70% free memory, 1.24 GiB swap in use, 9% root disk
  use, and 18% use on `/Volumes/Betty`.

No environment has material memory or disk pressure blocking qualification.

## Cleanup And Disposition

- Elm's obsolete `.before-shadow-align` extension backups were removed. Its
  installed extension directory now contains the expected 0.2.5 SQL and control
  files.
- Mac's installed historical version and upgrade SQL scripts are expected
  extension upgrade support, not stale rollout artifacts.
- Temporary Azure storage account `ii42deploy20260827` was deleted from
  `ii-commons-east-group` and a subsequent account listing returned no match.
- The Azure runtime fleet was deliberately retained, but Shadow no longer
  dispatches to it because `ii42.runtime_accelerators=[]` after bulk rebuilds.

The Mac qualification scope is the declared `postgres` target. Historical
benchmark databases with older extension catalogs and a separate alpha test
cluster under `/Volumes/Betty/Tmp/ii42-alpha-mass-pubmed-full/data` are outside
this rollout and were left untouched. These do not block the target's
qualification.

Shadow, Elm, and Mac have passed the bounded v0.2.5 current-only rollout gates.
The rollout monitor can be paused.
