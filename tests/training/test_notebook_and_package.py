from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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


def _load_builder():
    path = Path("scripts/build_xlmr_colab_notebook.py").resolve()
    spec = importlib.util.spec_from_file_location("promptsec_colab_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checked_notebook():
    path = Path("notebooks/PromptSec_FM_XLMR_Multitask_Colab.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _sources(notebook):
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    return markdown, code


def test_checked_in_notebook_is_valid_and_has_exact_section_order() -> None:
    builder = _load_builder()
    notebook = _checked_notebook()
    headings = [
        cell.source.splitlines()[0].removeprefix("## ")
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## ")
    ]
    assert headings == list(builder.SECTION_TITLES)
    assert len(notebook.cells) == 51


def test_notebook_configuration_is_public_github_first_and_full_run_is_guarded() -> None:
    notebook = _checked_notebook()
    _, code = _sources(notebook)
    configuration = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "GITHUB_OWNER" in cell.source
    )
    assert 'GITHUB_OWNER = "LIGHTER91"' in configuration
    assert 'GITHUB_REPOSITORY = "PromptSec-FM"' in configuration
    assert 'GITHUB_REF = "main"' in configuration
    assert 'REPO_DIR = "/content/PromptSec-FM"' in configuration
    assert 'DRIVE_ROOT = "/content/drive/MyDrive/PromptSec-FM"' in configuration
    assert "START_FULL_TRAINING = False" in configuration
    assert "RUN_SMOKE_TEST_FIRST = True" in configuration
    assert "https://github.com/{GITHUB_OWNER}" in configuration
    assert 'git", "clone' in code or '"clone"' in code
    assert '"--branch"' in code and '"--single-branch"' in code
    assert '"merge", "--ff-only"' in code
    assert "FORCE_RECLONE" in code
    assert "shell=False" in code


def test_notebook_drive_sha_and_safe_local_extraction_contract() -> None:
    notebook = _checked_notebook()
    _, code = _sources(notebook)
    assert "/content/drive/MyDrive/PromptSec-FM" in code
    assert "policybench-codex-v0.1.zip.sha256" not in code
    assert 'DATA_SHA256_FILE = DATA_ARCHIVE + ".sha256"' in code
    assert "/content/promptsec_data" in code
    assert 'iter(lambda: stream.read(chunk_size), b"")' in code
    assert "EXPECTED_ARCHIVE_SHA256" in code
    assert "relative.is_absolute()" in code
    assert '".." in relative.parts' in code
    assert "stat.S_ISLNK" in code
    assert "ZIP-slip" in code
    assert "compatible_local_extraction" in code
    assert "archive.extractall(local_root)" in code
    assert "read_bytes()" not in code


def test_notebook_installs_clone_and_only_orchestrates_training_cli() -> None:
    notebook = _checked_notebook()
    markdown, code = _sources(notebook)
    assert "pyproject.toml" in code
    assert '"training" not in optional_groups' in code
    assert '"pip"' in code and '"install"' in code and '"-e"' in code
    assert "editable_target" in code
    assert "expected_source_text = str(expected_source)" in code
    assert "sys.path.insert(0, expected_source_text)" in code
    assert "importlib.invalidate_caches()" in code
    assert "promptsec.training" in code
    assert code.index("sys.path.insert(0, expected_source_text)") < code.index(
        'importlib.import_module("promptsec.training")'
    )
    assert code.count("scripts/train_xlmr_multitask.py") >= 3
    assert '"--smoke-test"' in code
    assert '"--no-resume"' in code
    assert '"--resume" if RESUME else "--no-resume"' in code
    assert 'Path(CHECKPOINT_ROOT) / "smoke-test"' in code
    assert 'Path(REPORT_ROOT) / "smoke-test"' in code
    for duplicated_training_implementation in (
        "optimizer.step(",
        "loss.backward(",
        "for epoch in range(",
        "torch.optim.AdamW(",
    ):
        assert duplicated_training_implementation not in code
    assert "SILVER_VALIDATED" in markdown
    assert "human-Gold" in markdown


def test_notebook_embeds_expected_integrity_counts_without_payloads_or_secrets() -> None:
    notebook = _checked_notebook()
    markdown, code = _sources(notebook)
    combined = markdown + "\n" + code
    for expected in (
        "train=1012",
        "validation=242",
        "test_policy_family_ood=284",
        "test_domain_ood=491",
        "test_language_ood=3000",
        "test_counterfactual=344",
        "human_review_candidates=627",
        'counterfactual_groups") != 720',
        'checksums_checked") != 17',
        'split_files_checked") != 7',
    ):
        assert expected in combined
    assert "train_test_split" not in combined
    assert "random_split" not in combined
    assert "OPENAI_API_KEY" not in combined
    assert "HF_TOKEN" not in combined
    assert re.search(r"\bsk-[A-Za-z0-9_-]{20,}", combined) is None
    assert "pb_banking_" not in combined
    assert "raw_generation" not in combined
    assert len(nbformat.writes(notebook)) < 150_000


def test_notebook_builder_is_deterministic() -> None:
    builder = _load_builder()
    first = builder.build_notebook()
    second = builder.build_notebook()
    nbformat.validate(first)
    nbformat.validate(second)
    assert nbformat.writes(first) == nbformat.writes(second)
    assert [cell.id for cell in first.cells] == [
        f"promptsec-{index:03d}" for index in range(len(first.cells))
    ]
