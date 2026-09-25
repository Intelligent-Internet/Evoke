# Security Policy

## Supported Versions

Security fixes are developed against the current extension version declared in
`ii42.control`. The package does not ship historical Evoke catalogs, payload
readers, or version-to-version upgrade SQL. Unsupported historical formats must
be rebuilt from their source relations. A qualified deployment of a compatible
current-format package can retain its indexes; stop PostgreSQL before replacing
a loaded shared-preload library and verify the catalog, binary, and model
contract together. For `psql_bm25s`, follow the side-by-side source-table
migration in [Migrating BM25 Indexes](docs/upgrading.md).

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private GitHub security-advisory workflow and include:

- the affected Evoke and PostgreSQL versions;
- operating system and architecture;
- the smallest reproducible SQL or deployment sequence;
- expected and observed behavior;
- whether untrusted database roles can trigger the issue;
- any crash, data-integrity, privilege, or information-disclosure impact.

Maintainers will acknowledge the report, reproduce it in an isolated
environment, and coordinate disclosure after a fix and regression test are
available. Never include production credentials, model secrets, or private
corpus data in a report.

## Model Checkout Trust

Treat a configured model checkout as executable server configuration. Its
directory, `manifest.json`, and artifacts must be owned and writable only by
the PostgreSQL deployment administrator. Evoke verifies declared artifact
digests and fails closed when they change, but the manifest is the trust root
that declares those digests. Do not place a production checkout in a path
writable by application roles or untrusted operating-system users.
