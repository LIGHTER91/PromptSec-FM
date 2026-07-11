# Local dataset workspace

Downloaded artifacts and intermediate datasets are intentionally not committed. The
audited PromptSec-Dataset v0.1 release is tracked so its reports, provenance, and
checksums can be reviewed.

The release is not cleared for publication while `licenses.json` reports
`BLOCKED_PENDING_LICENSE_REVIEW`.

```text
data/raw/<source>/<revision>/       Immutable upstream artifacts
data/processed/v0.1/               Canonical validated JSONL records
data/releases/promptsec-dataset-v0.1/  Audited experimental release
```

Never edit a raw artifact in place. Source revisions, URLs, licenses, record-level
provenance, and checksums are recorded by the configuration and manifest files.
