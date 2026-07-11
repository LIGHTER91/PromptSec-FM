"""Validated, atomic JSONL dataset construction."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_file
from promptsec.data.importers.base import BaseImporter
from promptsec.data.validation import require_valid_record


class PipelineError(RuntimeError):
    """Raised when a build cannot produce a complete validated output."""


@dataclass(frozen=True, slots=True)
class BuildReport:
    source_id: str
    source_revision: str | None
    imported_at: str
    output: str
    output_sha256: str
    records: int
    mapping_statuses: dict[str, int]
    verdicts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset(
    importer: BaseImporter,
    inputs: Mapping[str, str | Path],
    output: str | Path,
    *,
    report_path: str | Path | None = None,
) -> BuildReport:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    record_count = 0
    seen_ids: set[str] = set()
    mapping_statuses: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in importer.records(inputs):
                require_valid_record(record)
                record_id = record["id"]
                if record_id in seen_ids:
                    raise PipelineError(f"duplicate canonical id: {record_id}")
                seen_ids.add(record_id)
                mapping_status = record["metadata"]["dataset_provenance"]["mapping"]["status"]
                verdict = record["derived"]["prompt_injection_verdict"]
                mapping_statuses[mapping_status] += 1
                verdicts[verdict] += 1
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
                record_count += 1
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    report = BuildReport(
        source_id=importer.config.id,
        source_revision=importer.config.revision,
        imported_at=importer.imported_at,
        output=output_path.as_posix(),
        output_sha256=sha256_file(output_path),
        records=record_count,
        mapping_statuses=dict(sorted(mapping_statuses.items())),
        verdicts=dict(sorted(verdicts.items())),
    )
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
