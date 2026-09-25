# Support

## Documentation And Questions

Start with the [documentation map](docs/README.md),
[getting-started guide](docs/getting-started.md), and
[API reference](docs/api-reference.md). When asking for help, include the Evoke
version, PostgreSQL major, operating system and architecture, relevant SQL,
and the complete error text.

Use repository discussions or the question channel provided by the hosting
project for usage and design questions. Keep reproducible product defects in
the issue tracker.

## Bug Reports

Before reporting a bug, reproduce it on a supported PostgreSQL version and a
current Evoke package or source checkout. Include a minimal schema and query,
`SELECT extversion FROM pg_extension WHERE extname = 'evoke'`, and whether the
index is BM25 or `sae = true`. Remove credentials, private data, model assets,
and infrastructure addresses.

## Security Reports

Do not report vulnerabilities in public issues. Follow
[SECURITY.md](SECURITY.md) for the supported-version and private-reporting
policy.

## Operational Boundaries

The maintainers cannot recover user data or validate an unqualified package,
model, or runtime combination. Back up the database and verify package,
dependency, model, and PostgreSQL-major fingerprints before deployment.
