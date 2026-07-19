from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import nbformat

from promptsec.data.hashing import sha256_file
from promptsec.training import colab_package


def _fake_release(root: Path) -> None:
    root.mkdir()
    for name in colab_package.RELEASE_FILES:
        if name.endswith(".jsonl"):
            value = '{"id":"redacted-fixture"}\n'
        elif name == "manifest.json":
            value = json.dumps({"records": 6000}) + "\n"
        elif name == "checksums.sha256":
            value = "metadata only\n"
        else:
            value = "{}\n"
        (root / name).write_text(value, encoding="utf-8", newline="\n")


def test_colab_package_is_minimal_deterministic_and_source_immutable(tmp_path, monkeypatch) -> None:
    source = tmp_path / "policybench-v0.1"
    _fake_release(source)
    before = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    monkeypatch.setattr(colab_package, "iter_dataset_records", lambda root: [{}] * 6000)
    monkeypatch.setattr(
        colab_package,
        "validate_release_directory",
        lambda root, records: {"validation_status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        colab_package,
        "validate_record_collection",
        lambda records: {
            "validation_status": "PASS",
            "records": 6000,
            "invalid_records": 0,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        colab_package,
        "release_file_hashes",
        lambda root: {
            path.name: sha256_file(path) for path in Path(root).iterdir() if path.is_file()
        },
    )
    first = colab_package.package_policybench_for_colab(
        source, tmp_path / "one" / "policybench.zip"
    )
    second = colab_package.package_policybench_for_colab(
        source, tmp_path / "two" / "policybench.zip"
    )
    assert first["archive_sha256"] == second["archive_sha256"]
    assert (
        first["archive_sha256"] == hashlib.sha256(Path(first["archive"]).read_bytes()).hexdigest()
    )
    assert before == {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    with zipfile.ZipFile(first["archive"]) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert any(name.endswith("/colab_input_manifest.json") for name in names)
        assert not any("raw" in name or "quarantine" in name for name in names)


def test_checked_in_notebook_is_valid_and_has_required_sections() -> None:
    path = Path("notebooks/PromptSec_FM_XLMR_Multitask_Colab.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for title in (
        "GPU and runtime preflight",
        "Dataset archive verification",
        "Release and split integrity checks",
        "Tokenization and serialization preview using redacted examples",
        "Optional small smoke test",
        "Full training",
        "Counterfactual evaluation",
        "Language analysis",
        "Hard-negative analysis",
        "Checkpoint export",
    ):
        assert title in markdown
    assert "FacebookAI/xlm-roberta-base" in code
    assert "TRAINING_MODE" in code
    assert "RESUME" in code
    assert "sha256" in code.lower()
