# Local dataset workspace

Downloaded and generated datasets are intentionally not committed.

```text
data/raw/<source>/<revision>/       Immutable upstream artifacts
data/processed/v0.1/               Canonical validated JSONL records
```

Never edit a raw artifact in place. Source revisions, URLs, licenses, record-level
provenance, and checksums are recorded by the configuration and manifest files.

