# Generated reports

Ad hoc build and validation reports written here are ignored by Git. The Phase 3.2
statistics and label-review queue are tracked at:

```text
reports/data_statistics/dataset_v0.1.{json,md}
reports/label_mapping/review_queue_v0.1.jsonl
```

These reports describe the full imported corpus before exact-duplicate removal. The
release-local reports under `data/releases/promptsec-dataset-v0.1/` additionally
record deduplication decisions, split assignments, licenses, and checksums.
