# Local dataset workspace

Downloaded artifacts, intermediate datasets, split JSONL files, and review queues are
intentionally not committed. A release directory exposes only redacted metadata that can
be reviewed without redistributing source-derived records.

`checksums.sha256` remains local and covers the complete reconstructed release.
`checksums_pipeline.txt` is publishable and covers only the allow-listed metadata files.
The dataset payload is not cleared for publication while `licenses.json` reports
`BLOCKED_PENDING_LICENSE_REVIEW` for that component.

```text
data/raw/<source>/<revision>/       Immutable upstream artifacts
data/processed/v0.1/               Canonical validated JSONL records
data/releases/promptsec-dataset-v0.1/  Local payload plus publishable redacted metadata
```

Never edit a raw artifact in place. Source revisions, URLs, licenses, record-level
provenance, and checksums are recorded by the configuration and manifest files.
