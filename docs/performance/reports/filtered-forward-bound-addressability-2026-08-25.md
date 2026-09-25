# Filtered Forward-Bound Addressability Audit

## Decision

A query-term membership/rank-select layer alone does not materially reduce
the cost of dispersed filtered search. It is rejected as the next product
format change. Coarsening the exact bound from b8 to b16 or b64 is also
rejected because the lower metadata cost is overwhelmed by extra competitive
row and posting work.

No route threshold, score, result, root, or lifecycle was changed by this
audit.

## Question

The current full PubMed `forward_bound` route decodes approximately 29.4
million query-term b8 entries, or 147 MB, before the filter removes irrelevant
blocks. The audit asked whether an adaptive representation could preserve the
same conservative bounds while reading only block memberships and the compact
bound windows intersected by the filter.

The representation oracle keeps the existing delta-varint plus float32 stream
for sparse terms. A term switches to a membership bitmap plus rank-addressed
float32 bounds only when that representation is smaller. Selected bounds are
charged at 4 KiB physical windows rather than ideal logical float reads.

## Method

The immutable Shadow 500,000-document PubMed root was inspected through an
isolated PostgreSQL 18 postmaster. One frozen 45-atom query and six filter
locality shapes were measured. The root identity was checked before and after
all probes. The audit consumed at most 52,224 KiB RSS and completed in 51.82
seconds.

The evidence is retained at:

```text
/data/evoke-builds/evoke-cq3-b64-oracle-1/
pubmed-500k-addressable-oracle.json
```

SHA-256:

```text
6f19f8b812ca1b6d077ee9dd1013016879b8f372dce112e1bec1db0c11040648
```

## Results

The table reports total projected query bytes: bound metadata or addressable
bound windows plus competitive forward rows. Ratios compare the addressable
projection with the current term stream at the same block size.

| Filter shape | b8 current | b8 addressable | Ratio | b16 addressable | b64 addressable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scattered 0.6% | 1.169 MB | 1.087 MB | 0.929 | 1.302 MB | 2.104 MB |
| Scattered 2.5% | 1.384 MB | 1.301 MB | 0.940 | 2.050 MB | 6.523 MB |
| Distributed runs 2.5% | 1.357 MB | 1.274 MB | 0.939 | 1.890 MB | 6.241 MB |
| Clustered 2.5% | 1.259 MB | 0.741 MB | 0.589 | 1.388 MB | 6.016 MB |
| Scattered 10% | 1.141 MB | 1.058 MB | 0.928 | 1.315 MB | 7.044 MB |
| Unfiltered | 1.179 MB | 1.096 MB | 0.930 | 1.463 MB | 9.152 MB |

Every level retained the exact top-50 containment property. Nevertheless,
none of the dispersed filters reached the required 25% reduction. Only the
clustered filter benefited materially.

For b8, 165,109 query-term bounds occupied 828,484 bytes. Only four query
terms selected dense membership, while 38 retained sparse streams and three
had no positive bound entries. The scattered 0.6% filter touched 113 physical
bound windows, so logical selectivity did not become physical I/O locality.

## Interpretation

The remaining cost is not caused by choosing b8 instead of b16/b64. It is
caused by two properties of the workload:

1. Most query terms remain sparse enough that a whole membership bitmap is not
   a smaller authority than the ordered stream.
2. Allowed documents are dispersed across the corpus, so even dense-term
   rank-select references touch nearly every physical bound window.

The 500K scale point therefore rejects a full-root rebuild for this shape.
Running the same representation on the 7.8M root would measure a larger
version of the same linear sparse-stream and scattered-window cost.

## Next Research Boundary

Future work must reduce actual sparse/high-DF term decoding before changing
the product. A valid proposal must first expose, for the same immutable root
and query/filter set:

- term-by-term bound-entry and page costs;
- allowed-block intersection operations and skipped physical pages;
- canonical posting and competitive-row bytes;
- exact top-k equality, cancellation, RSS, and concurrency behavior.

Acceptable structural candidates include a skip-capable block posting layout
or another same-root block-max representation whose oracle materially reduces
decoded entries and pages. Another block-size grid, global route factor,
cache size, or rank-select-only wrapper is explicitly out of scope.
