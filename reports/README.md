# Generated reports

Ad hoc build and validation reports written here are ignored by Git. Aggregate Phase 3.2
statistics are tracked at:

```text
reports/data_statistics/dataset_v0.1.{json,md}
```

Label-review queues under `reports/label_mapping/` contain source-derived text and always
remain local and ignored. Aggregate reports describe the full imported corpus before
exact-duplicate removal. Release-local public reports additionally record redacted
deduplication decisions, split assignments, licenses, and metadata-only checksums.
