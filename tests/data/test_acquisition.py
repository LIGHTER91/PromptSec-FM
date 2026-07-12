from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import promptsec.data.acquisition as acquisition_module
from promptsec.data.acquisition import AcquisitionError, acquire_source, acquire_sources
from promptsec.data.config import SourceConfig


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_config(
    tmp_path: Path,
    *,
    digest: str,
    revision: str = "1" * 40,
    method: str = "http_artifacts",
    local_path: str | None = None,
) -> SourceConfig:
    local_path_line = f'local_path = "{local_path}"\n' if local_path else ""
    snapshot_fields = ""
    if method == "python_package_snapshot":
        snapshot_fields = """
package_name = "agentdojo"
package_version = "0.1.35"
benchmark_version = "v1.2.2"
snapshot_filename = "definitions.snapshot.json"
"""
    path = tmp_path / f"{method}.toml"
    path.write_text(
        f"""
schema_version = "0.1"

[source]
id = "synthetic"
name = "Synthetic"
homepage = "https://example.invalid/"
repository = "https://example.invalid/repository.git"
version = "1.0"
revision = "{revision}"
license_manifest = "manifests/synthetic.json"
importer = "tests.synthetic:Importer"

[acquisition]
method = "{method}"
cache_path = "data/raw/synthetic/{revision}"
used_files = ["payload.txt"]
{snapshot_fields}

[scenario]
language = "en"
delivery_mode = "DIRECT"
source_role = "USER"
content_origin = "USER_AUTHORED"
ingestion_path = "CHAT_INPUT"
modality = "TEXT"
source_integrity = "UNKNOWN"

[fields]
text = ["text"]
labels = ["label"]
record_id = ["id"]

[[artifacts]]
id = "payload"
split = "all"
format = "text"
url = "https://example.invalid/payload.txt"
sha256 = "{digest}"
{local_path_line}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return SourceConfig.load(path)


def test_offline_acquisition_reuses_a_checksum_verified_cache(tmp_path: Path) -> None:
    payload = b"pinned payload\n"
    config = _source_config(tmp_path, digest=_digest(payload))
    target = tmp_path / "raw" / config.id / config.revision / "payload.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    result = acquire_source(config, tmp_path / "raw", offline=True)

    assert result.artifacts == {"payload": target}
    assert result.snapshot is None


def test_offline_acquisition_fails_when_cache_is_missing(tmp_path: Path) -> None:
    config = _source_config(tmp_path, digest=_digest(b"expected"))

    with pytest.raises(AcquisitionError, match="offline mode: missing pinned artifact"):
        acquire_source(config, tmp_path / "raw", offline=True)


def test_local_file_override_still_enforces_checksum_pin(tmp_path: Path) -> None:
    payload = b"expected"
    config = _source_config(tmp_path, digest=_digest(payload))
    override = tmp_path / "payload.txt"
    override.write_bytes(payload)

    result = acquire_source(
        config,
        tmp_path / "unused-cache",
        offline=True,
        local_path=override,
    )
    assert result.artifacts == {"payload": override.resolve()}

    override.write_bytes(b"tampered")
    with pytest.raises(AcquisitionError, match="checksum .* expected"):
        acquire_source(
            config,
            tmp_path / "unused-cache",
            offline=True,
            local_path=override,
        )


def test_local_checkout_override_rejects_revision_pin_mismatch(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "PromptSec tests"],
        check=True,
    )
    payload = checkout / "payload.txt"
    payload.write_bytes(b"pinned")
    subprocess.run(["git", "-C", str(checkout), "add", "payload.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "fixture"], check=True)
    config = _source_config(
        tmp_path,
        digest=_digest(payload.read_bytes()),
        revision="f" * 40,
        local_path="payload.txt",
    )

    with pytest.raises(AcquisitionError, match="does not match pinned revision"):
        acquire_source(config, tmp_path / "raw", local_path=checkout)


def test_offline_snapshot_acquisition_never_exports_missing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"wheel"
    config = _source_config(
        tmp_path,
        digest=_digest(wheel),
        method="python_package_snapshot",
    )
    override = tmp_path / "package.whl"
    override.write_bytes(wheel)
    exports: list[Path] = []
    monkeypatch.setattr(
        acquisition_module,
        "export_snapshot",
        lambda output, *args, **kwargs: exports.append(Path(output)),
    )

    with pytest.raises(AcquisitionError, match="offline mode: missing pinned snapshot"):
        acquire_source(config, tmp_path / "raw", offline=True, local_path=override)
    assert exports == []


def test_online_snapshot_acquisition_exports_then_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = b"wheel"
    config = _source_config(
        tmp_path,
        digest=_digest(wheel),
        method="python_package_snapshot",
    )
    override = tmp_path / "package.whl"
    override.write_bytes(wheel)
    validated: list[Path] = []

    def fake_export(output: Path, *args: object, **kwargs: object) -> None:
        Path(output).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(acquisition_module, "export_snapshot", fake_export)
    monkeypatch.setattr(
        acquisition_module,
        "validate_snapshot",
        lambda path, source: validated.append(Path(path)),
    )

    result = acquire_source(config, tmp_path / "raw", local_path=override)

    assert result.snapshot == tmp_path / "definitions.snapshot.json"
    assert validated == [result.snapshot]


def test_acquire_sources_rejects_unknown_selected_source_and_override(tmp_path: Path) -> None:
    config = _source_config(tmp_path, digest=_digest(b"payload"))

    with pytest.raises(AcquisitionError, match="not configured"):
        acquire_sources([config], tmp_path / "raw", source_ids=["missing"])
    with pytest.raises(AcquisitionError, match="do not match selected"):
        acquire_sources(
            [config],
            tmp_path / "raw",
            source_ids=[config.id],
            local_paths={"missing": tmp_path},
        )
