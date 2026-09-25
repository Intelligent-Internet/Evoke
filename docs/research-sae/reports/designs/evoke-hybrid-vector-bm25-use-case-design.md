# Hybrid Vector/BM25 Search Use-Case Design

> Historical design note. The current product route uses one model-backed
> unified posting index and `ii42_query(...)`. The late-fusion engine remains
> an owner-only diagnostic surface, not an application route.

This document records the first concrete use-case design for hybrid
Vector/BM25 retrieval in the current Commons deployment.

The goal is not to describe every possible hybrid strategy. The goal is to
describe the workload we actually have, the shape of the current data, and the
first design that is both operationally realistic and easy to validate.

This document intentionally focuses on the typical document-level case where
the main serving table already carries both BM25 fields and a document vector.
Corpus-specific shapes that require chunk-to-document aggregation are useful,
but they are not the generic design this note is trying to capture.

## Current Database Layout

The live retrieval design for the generic document-search path currently
centers on two corpora:

1. `commons.data_arxiv`
2. `commons.data_pubmed`

The serving pattern is:

- the main table is responsible for finding the right documents
- BM25 handles exact lexical relevance
- vector similarity handles semantic recall

## Rough Scale

The current live corpus scale on `elm` is approximately:

- `commons.data_arxiv`: about `2.85M` rows
- `commons.data_pubmed`: about `7.06M` rows

These numbers matter because a design that is acceptable on a small dataset
can be completely wrong once candidate generation reaches the million-row
range.

## Key Search Fields

### arXiv

`commons.data_arxiv` currently exposes the important retrieval fields:

- `title text`
- `abstract text`
- `authors text[]`
- `categories text[]`
- `organizations text`
- `vector halfvec`

Active retrieval indexes:

- `data_arxiv__title_abstract__field_aware_bm25_idx`
- `data_arxiv__vector__vchord_idx`

Supporting filter indexes include:

- `data_arxiv__categories__gin_idx`
- `data_arxiv__organizations__trgm_idx`

### PubMed

`commons.data_pubmed` currently exposes:

- `title text`
- `abstract text`
- `authors text[]`
- `journal_title citext`
- `categories text[]`
- `vector halfvec`

Active retrieval indexes:

- `data_pubmed__title_abstract__field_aware_bm25_idx`
- `data_pubmed__vector__vchord_idx`

## The Query We Actually Need

The real workload is not:

- pure BM25
- pure vector top-k
- or filter-only search

The real query shape is:

1. apply some relational constraints
2. retrieve semantically relevant candidates using vector search
3. retrieve lexically exact candidates using BM25
4. rank once using a fused score
5. return the final top-k rows from the main serving table

Typical filters include:

- `authors`
- `journal_title`
- `categories`
- time ranges
- source-specific metadata such as `organizations`

The key product requirement is that vector should remain the primary relevance
signal. BM25 is important, but BM25 is not the dominant ranking source for
these AI retrieval workloads.

## Why Planner-Driven Single-SQL Is Not Enough

A naive query shape often looks like this:

```sql
SELECT *
FROM docs
WHERE ...
ORDER BY vector <-> query_vector
LIMIT k;
```

That shape leaves too much to the PostgreSQL planner.

For this workload, that is dangerous because:

- the planner may choose a filter-first path
- the vector ANN index may not be used
- the result may become an exact rerank over a small filtered set
- the product intent of vector-first retrieval is lost

This is acceptable for some analytical queries. It is not the right default
for our serving path.

## Why Single-Source Candidate Generation Is Not Enough

A first reaction is often:

- fetch `LIMIT 1000` from vector
- apply filters
- then rerank

That is still not good enough.

If the filter is hard:

- `1000` may be too small
- the post-filter set may be empty or too thin

If the filter is weak:

- `1000` may be wasteful

If BM25 is not consulted at all:

- lexically precise hits may disappear

So the correct first design is not "vector only, then filter". It is a pooled
candidate design.

## V1 Candidate Strategy

The first design uses two candidate sources:

1. vector candidates
2. BM25 candidates

Then:

3. union the document identities
4. apply relational filters on the pooled candidates
5. compute the final hybrid score once
6. return final top-k

This has three immediate benefits:

- vector retains an explicit candidate path
- BM25 can recover exact lexical hits
- filters operate on a much smaller working set than the whole table

The first SQL prototype keeps candidate budgets fixed only because that makes
validation easier. Adaptive expansion is the natural next step.

## Why Raw L2 Should Not Be Added To BM25

In the current Commons embedding path, query vectors are:

- prefixed with `query: `
- truncated to 256 dimensions
- L2-normalized

Under that setup, the vectors lie on the unit sphere. For unit vectors:

```text
||u - v||^2 = 2 - 2 cos(u, v)
cos(u, v) = 1 - (l2^2 / 2)
```

That means the current L2 distance is really a monotonic transform of cosine
similarity. It is not an arbitrary Euclidean magnitude.

So:

- `l2` is lower-is-better
- BM25 is higher-is-better
- direct addition is meaningless

The first correction is to map L2 back into a cosine-like similarity space.

## Why Raw BM25 Should Not Be Treated As Global

BM25 raw score is meaningful inside one query result set, but it is not
globally calibrated across queries.

Different queries produce different score ranges because of:

- token rarity
- query length
- field hit structure
- query normalization options

So a score-fusion design should treat BM25 as a query-local signal unless a
stronger calibration layer is introduced.

## V1 Score Formula

The first design used:

```text
vector_sim = clamp(1 - (l2^2 / 2), 0, 1)
bm25_norm = bm25_score / max_bm25_score_in_pool
hybrid_score = sqrt(vector_sim) + 0.6 * bm25_norm
```

This has a deliberate bias:

- vector remains the primary relevance signal
- BM25 acts as a lexical precision boost
- a document can still rank from strong semantic evidence alone
- BM25 helps exact terminology without overwhelming semantic relevance

The `sqrt(vector_sim)` transform is intentionally mild. It widens separation
between stronger semantic hits while keeping the score easy to inspect.

This V1 shape was good enough to validate candidate generation and score
normalization. It was not yet the final serving formula.

## Why Pool-Local BM25 Normalization

The normalization denominator is the maximum BM25 score in the filtered
candidate pool, not the maximum BM25 score in the original unfiltered BM25
candidate set.

That matters because the final ranking is over the filtered pool. If
normalization is computed before filtering, the scale can be set by rows that
are not even eligible for the final answer.

So the more correct V1 semantics are:

- generate candidates
- filter pooled candidates
- normalize BM25 across the rows that still matter
- compute the hybrid score once

## V2 Final Score Shape

The current settled design in `commons-dev` keeps the same candidate-pool
strategy but upgrades the scoring shape in two ways:

1. add vector confidence gating
2. optionally add a bounded recency prior after semantic normalization

The current vector gate is:

```text
if l2 <= 0.95: vector_conf = 1
if l2 >= 1.18: vector_conf = 0
otherwise:     linear decay from 1 to 0
```

The current semantic and final score shape is:

```text
vector_sim = clamp(1 - (l2^2 / 2), 0, 1)
vector_component = vector_conf * sqrt(vector_sim)

bm25_norm = bm25_score / max_bm25_score_in_pool
bm25_component = 0.6 * bm25_norm

semantic_score = (vector_component + bm25_component) / 1.6
final_score = semantic_weight * semantic_score
            + recency_weight * recency_score
```

This solves a real weakness in V1.

V1 aligned score scales, but it still let weak semantic matches keep a small
positive contribution even when the distance had already drifted into the
measured background-overlap region.

The V2 gate makes the semantic contribution behave more like evidence:

- strong semantic neighbors keep full weight
- borderline neighbors decay smoothly
- background-distance rows stop contributing semantic score

The semantic normalization step is also important. Once
`vector_component + bm25_component` is divided by `1.6`, the optional recency
prior can be added as a bounded tie-breaker instead of competing with an
unnormalized hybrid magnitude.

## Recency Prior

The current serving design treats recency as an optional bounded prior, not as
the main ranking signal.

The generic shape is:

```text
recency_score = linear_decay(published_on, window_days)
```

where:

- newly published rows are close to `1`
- rows outside the time window decay to `0`
- missing dates contribute `0`

This is only appropriate for corpora where "slightly newer is usually better"
is a defensible product prior.

The current `commons-dev` choices are:

- arXiv:
  - `semantic_weight = 0.90`
  - `recency_weight = 0.10`
  - `window = 5 years`
- PubMed:
  - `semantic_weight = 0.95`
  - `recency_weight = 0.05`
  - `window = 5 years`

These are intentionally conservative.

- arXiv gets a stronger recency bias because new preprints often supersede
  older drafts quickly.
- PubMed gets a weaker bias because older landmark papers remain important for
  much longer.

So the design rule is:

- keep recency bounded
- keep semantic relevance dominant
- use metadata-aware date normalization before computing any time prior

For example, PubMed date metadata is often only year-level or year-month
level. In the current serving implementation, that coarse metadata is first
mapped to a representative date before computing the recency prior, so the
system does not accidentally over-reward rows just because their date field is
imprecise.

## arXiv Validation

The first validation target was `commons.data_arxiv`.

This corpus is a good first target because:

- its main serving table already has both BM25 and vector indexes
- the BM25 index is not stale
- the main filter columns already have supporting indexes
- relevance is easy to inspect manually

The first test used:

- seed vector from `2311.03739`
- BM25 query text `automated proof synthesis rust`
- field-aware BM25 weights:
  - `title = 2.0`
  - `abstract = 1.0`
- relational filters:
  - `categories @> ARRAY['cs.AI']`
  - `organizations ILIKE '%microsoft research%'`

Observed top results were semantically correct:

1. `2311.03739` `Leveraging Large Language Models for Automated Proof Synthesis in Rust`
2. `2409.13082` `AutoVerus: Automated Proof Generation for Rust Code`
3. `2410.15756` `Automated Proof Generation for Rust Code via Self-Evolution`

This is exactly the kind of mixed lexical/semantic neighborhood we want.

## Efficiency Notes From The First Run

Two execution details matter:

1. the vector candidate branch must use an external query vector constant
2. PostgreSQL JIT can dominate total latency for this query shape

When the vector branch was written in a way that let planner treat the query
vector as a row-derived expression, the engine fell back to a table scan on
the vector side. That shape is not acceptable for production use.

Once the query vector was supplied as a constant client variable, the vector
branch correctly used `data_arxiv__vector__vchord_idx`.

Measured on the corrected shape:

- with JIT on: about `1.88s`
- with `jit = off`: about `163 ms`

That means the retrieval design itself is already in a useful latency range.
The long version was mostly PostgreSQL JIT overhead, not candidate generation
or score fusion cost.

## What This V1 Solves

This design solves the first practical problem:

- one request
- explicit vector path
- explicit BM25 path
- pooled candidates
- relational filters
- final fused top-k ordering

It gives us a concrete baseline for:

- result quality review
- candidate budget tuning
- score-shape review
- later API design

## Reference SQL Prototype

The current arXiv-style prototype below matches the settled V2 design more
closely than the earlier V1 sketch. It keeps vector candidates, BM25
candidates, pooled identities, filter application, semantic scoring, and the
bounded recency prior visible in one statement.

The query expects a client-side `psql` variable named `query_vector`.

```sql
WITH params AS (
    SELECT
        'automated proof synthesis rust'::text AS bm25_query,
        ARRAY['title', 'abstract']::text[] AS field_names,
        ARRAY[2.0, 1.0]::real[] AS field_weights,
        300::int4 AS vector_candidate_k,
        300::int4 AS bm25_candidate_k,
        20::int4 AS final_k,
        0.90::double precision AS semantic_weight,
        0.10::double precision AS recency_weight,
        1825::int4 AS recency_window_days
),
vector_candidates AS (
    SELECT
        a.ctid,
        a.id,
        (a.vector <-> (:'query_vector')::halfvec(256))::real AS distance
    FROM commons.data_arxiv AS a
    CROSS JOIN params AS p
    WHERE a.vector IS NOT NULL
    ORDER BY a.vector <-> (:'query_vector')::halfvec(256)
    LIMIT (
        SELECT vector_candidate_k
        FROM params
    )
),
bm25_hits AS (
    SELECT h.ctid, h.score
    FROM params AS p
    CROSS JOIN LATERAL ii42_field_aware_query(
        'commons.data_arxiv__title_abstract__field_aware_bm25_idx'::regclass,
        p.bm25_query,
        p.field_names,
        p.field_weights,
        p.bm25_candidate_k
    ) AS h
),
bm25_candidates AS (
    SELECT
        a.ctid,
        a.id,
        h.score
    FROM bm25_hits AS h
    JOIN commons.data_arxiv AS a
      ON a.ctid = h.ctid
),
candidate_pool AS (
    SELECT ctid, id
    FROM vector_candidates
    UNION
    SELECT ctid, id
    FROM bm25_candidates
),
filtered_docs AS (
    SELECT
        a.ctid,
        a.id,
        a.title,
        a.categories,
        a.organizations
    FROM candidate_pool AS p
    JOIN commons.data_arxiv AS a
      ON a.id = p.id
    WHERE a.categories @> ARRAY['cs.AI']::text[]
      AND a.organizations ILIKE '%microsoft research%'
),
pool_max AS (
    SELECT COALESCE(MAX(b.score), 0.0::real) AS max_bm25_score
    FROM filtered_docs AS d
    LEFT JOIN bm25_candidates AS b
      ON b.id = d.id
),
scored AS (
    SELECT
        d.id,
        d.title,
        d.categories,
        d.organizations,
        d.publish_date,
        v.distance,
        b.score AS bm25_score,
        GREATEST(
            0.0::double precision,
            1.0::double precision
                - ((v.distance::double precision * v.distance::double precision)
                / 2.0::double precision)
        ) AS vector_sim,
        CASE
            WHEN m.max_bm25_score > 0.0::real
                THEN b.score / m.max_bm25_score
            ELSE 0.0::real
        END AS bm25_norm,
        CASE
            WHEN v.distance IS NULL THEN 0.0::double precision
            WHEN v.distance <= 0.95::real THEN 1.0::double precision
            WHEN v.distance >= 1.18::real THEN 0.0::double precision
            ELSE ((1.18::double precision - v.distance::double precision)
                / 0.23::double precision)
        END AS vector_conf
    FROM filtered_docs AS d
    LEFT JOIN vector_candidates AS v
      ON v.id = d.id
    LEFT JOIN bm25_candidates AS b
      ON b.id = d.id
    CROSS JOIN pool_max AS m
),
components AS (
    SELECT
        s.*,
        COALESCE(
            s.vector_conf * SQRT(s.vector_sim),
            0.0::double precision
        ) AS vector_component,
        0.6::double precision
            * COALESCE(s.bm25_norm::double precision, 0.0::double precision)
            AS bm25_component,
        CASE
            WHEN s.publish_date IS NULL OR s.publish_date <= 0
                THEN 0.0::double precision
            ELSE GREATEST(
                0.0::double precision,
                1.0::double precision - (
                    LEAST(
                        GREATEST(
                            (
                                CURRENT_DATE
                                - TO_DATE(
                                    LPAD(s.publish_date::text, 8, '0'),
                                    'YYYYMMDD'
                                )
                            )::double precision,
                            0.0::double precision
                        ),
                        (SELECT recency_window_days FROM params)::double precision
                    ) / (SELECT recency_window_days FROM params)::double precision
                )
            )
        END AS recency_score
)
SELECT
    c.id,
    c.title,
    c.distance,
    c.bm25_score,
    c.vector_sim,
    c.vector_conf,
    c.bm25_norm,
    c.vector_component,
    c.bm25_component,
    (
        (
            c.vector_component + c.bm25_component
        ) / 1.6::double precision
    ) AS semantic_score,
    c.recency_score,
    (
        (SELECT semantic_weight FROM params)::double precision
            * (
                (
                    c.vector_component + c.bm25_component
                ) / 1.6::double precision
            )
        + (SELECT recency_weight FROM params)::double precision
            * c.recency_score
    ) AS fusion_score
FROM components AS c
ORDER BY fusion_score DESC, c.id
LIMIT (
    SELECT final_k
    FROM params
);
```

Suggested `psql` session shape:

```sql
\set query_vector [0.1,0.2,...]
SET LOCAL jit = off;
SET LOCAL vchordrq.probes = 100;
```

## First Calibration Pass

The first arXiv calibration pass sampled five seeds:

- `2311.03739`
- `2409.13082`
- `2410.15756`
- `2403.15122`
- `2309.12938`

Observed nearest-neighbor distances stayed roughly in:

- `0.70` to `0.88`

Observed sampled background `p05` values stayed roughly in:

- `1.18` to `1.21`

That creates a useful practical separation:

- a clear semantic neighborhood below about `0.95`
- a likely background region at about `1.18` and beyond

This suggests a conservative first gate:

```text
if l2 <= 0.95: vector_conf = 1
if l2 >= 1.18: vector_conf = 0
otherwise:     linear decay from 1 to 0
```

This is intentionally conservative. It avoids suppressing clearly relevant
neighbors while still removing vector contribution once the candidate reaches
the measured background overlap band.

## First V1/V2 Comparison

The first comparison kept the same candidate strategy and changed only the
vector component:

```text
V1: vector_component = sqrt(vector_sim)
V2: vector_component = vector_conf * sqrt(vector_sim)
```

The BM25 side remained unchanged in this pass.

### Case 1: `2311.03739` / `automated proof synthesis rust` / `cs.AI`

For this query, the top vector neighbors were already inside the high-confidence
distance band. As expected, V1 and V2 produced the same top ordering.

That is a good sign: the gate is not disturbing clearly relevant semantic
neighbors.

### Case 2: `2403.15122` / `rust verification` / `cs.PL`

This case exposed the intended behavior.

Under V1, some rows with strong BM25 support but weaker vector distance still
ranked high because the vector component remained positive even once the
distance moved toward the background overlap region.

Under V2:

- rows around `l2 ~= 0.96` to `1.02` lost part of their vector contribution
- rows with stronger vector evidence moved upward
- pure lexical support no longer received extra help from a weak semantic match

This is exactly the behavior the confidence gate is supposed to enforce.

### Case 3: `2309.12938` / `code quality llm` / `cs.AI`

This case again showed no meaningful change in the top results because the
leading semantic neighbors were already close enough to stay inside the
full-confidence region.

That means the gate is selective rather than globally punitive.

### Expanded arXiv comparison set

The comparison set was then expanded to five arXiv cases:

- `2311.03739` / `automated proof synthesis rust` / `cs.AI`
- `2409.13082` / `rust code verification` / `cs.AI`
- `2410.15756` / `automated proof generation rust` / `cs.AI`
- `2403.15122` / `rust verification` / `cs.PL`
- `2309.12938` / `code quality llm` / `cs.AI`

Observed top-8 overlap:

- four cases: `8 / 8` overlap between V1 and V2
- one case: `6 / 8` overlap, with `2` rows dropped and `2` rows added

The one case that changed was the `rust verification / cs.PL` query, which is
exactly the kind of borderline semantic situation the gate is supposed to
affect.

This is the desired behavior:

- no change for clearly strong semantic neighborhoods
- selective change for rows whose vector distance is already drifting toward
  the measured background overlap band

On one same-shape `category-only` arXiv benchmark, the V1 and V2 SQL forms
also stayed in the same latency range:

- V1: about `117 ms`
- V2: about `109 ms`

That is not enough to claim a final benchmark win, but it is enough to say the
extra vector gate did not introduce a visible regression in this candidate-pool
design.

### PubMed distance check

The next calibration question was whether the arXiv-derived vector gate is
completely corpus-specific or whether PubMed shows a similar distance shape.

A first PubMed pass sampled five random seed articles with non-null vectors.
The observed values were again consistent:

- strong nearest-neighbor distances: about `0.61` to `0.83`
- sampled background `p05`: about `1.13` to `1.22`
- sampled background medians: about `1.27` to `1.33`

That does not prove a universal threshold, but it is strong evidence that the
current provisional gate is not obviously overfit to arXiv.

In particular:

- the semantic neighborhood still sits well below `0.95`
- the likely background overlap region still begins around `1.15+`

So the current `0.95 / 1.18` provisional gate remains a reasonable first
cross-corpus starting point.

## Tracking-Friendly Design

One lesson from the implementation work is that hybrid ranking is much easier
to debug when the query can expose its internal score components directly.

The practical tracking design should therefore treat the following values as
first-class diagnostics:

- `vector_distance`
- `bm25_score`
- `vector_sim`
- `vector_conf`
- `bm25_norm`
- `vector_component`
- `bm25_component`
- `semantic_score`
- `recency_score`
- `fusion_score`

This does not mean every production query should always return every
diagnostic. It means the SQL shape should make these components easy to expose
in a debug mode without changing the ranking logic.

### Recommended Debug Output

The simplest design is:

1. keep the serving query unchanged
2. add a debug variant that returns the score components above
3. make the final ordering still depend only on `fusion_score`

That gives three operational benefits:

- score regressions become inspectable row by row
- candidate-budget changes can be reviewed without guessing
- corpus-specific recency settings can be evaluated from real query output

### Recommended Sampling / Audit Shape

If query-level tracking is needed beyond interactive debugging, a lightweight
trace record should capture:

- query text
- corpus
- candidate budgets
- top-k ids
- top-k score components
- timing metadata

The key rule is to track enough to explain a ranking change, but not so much
that the trace path becomes a second serving system.

So the intended tracking design is:

- no change to the core ranking formula
- a debug/output surface that exposes score components cleanly
- optional sampled audit rows for regression review

## Interim Conclusion

The current document-level hybrid design has converged enough to treat V2 as
the working baseline:

- candidate pooling stays the same
- vector confidence gating is part of the settled formula
- BM25 remains pool-local and lightly weighted
- recency is a bounded optional prior, not a replacement for semantic ranking

The remaining open work is no longer "what is V2?".
The remaining open work is:

- candidate-budget tuning
- whether BM25 tail gating is worth the added complexity
- how much recency bias each corpus should actually keep
- how to expose score internals cleanly for tracking and regression review

## Related Files

- [Hybrid Vector/BM25 Search](../../../hybrid-search.md)
- [Hybrid Fusion Engine](../../../hybrid-fusion-engine.md)
