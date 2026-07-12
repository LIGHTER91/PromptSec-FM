from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from promptsec.data.importers import AgentDojoImporter, InjecAgentImporter

ROOT = Path(__file__).resolve().parents[2]


def test_v02_wheel_metadata_registers_agentic_importers_and_static_assets() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.2.0"
    assert project["project"]["optional-dependencies"]["agentdojo-snapshot"] == [
        "agentdojo==0.1.35"
    ]
    assert project["project"]["scripts"]["promptsec-dataset"] == "promptsec.data.cli:main"
    assert AgentDojoImporter.__name__ == "AgentDojoImporter"
    assert InjecAgentImporter.__name__ == "InjecAgentImporter"

    data_files = project["tool"]["setuptools"]["data-files"]
    expected = {
        "share/promptsec-dataset/configs": ["configs/*.yaml"],
        "share/promptsec-dataset/configs/sources": ["configs/sources/*.toml"],
        "share/promptsec-dataset/manifests": ["manifests/*.json"],
        "share/promptsec-dataset/manifests/sources": ["manifests/sources/*.json"],
        "share/promptsec-dataset/schemas": ["schemas/*.json"],
    }
    assert all(
        data_files.get(destination) == patterns for destination, patterns in expected.items()
    )
    assert not any(
        pattern.startswith(("data/", "reports/", "tests/"))
        for patterns in data_files.values()
        for pattern in patterns
    )

    packaged_paths: set[str] = set()
    for patterns in data_files.values():
        for pattern in patterns:
            matches = [path for path in ROOT.glob(pattern) if path.is_file()]
            assert matches, f"wheel data pattern has no matches: {pattern}"
            packaged_paths.update(path.relative_to(ROOT).as_posix() for path in matches)

    assert "configs/sources/agentdojo.toml" in packaged_paths
    assert "configs/sources/injecagent.toml" in packaged_paths
    assert "manifests/source-lock.json" in packaged_paths
    assert "schemas/promptsec-annotation-v1.schema.json" in packaged_paths
    assert not any(path.startswith(("data/", "reports/", "tests/")) for path in packaged_paths)


def test_frozen_taxonomy_v1_files_have_not_changed() -> None:
    expected_sha256 = {
        "docs/annotation_guidelines.md": (
            "f19114d2e21d56c102fc7ea85bb5e571e9bce3f0cb905a53e0403d52bcf1f400"
        ),
        "docs/taxonomy.md": ("4a6a5d0ff3f82d42835d14fe4d26df1d3aa0f9ed3c31696772326a0bd549804f"),
        "docs/taxonomy_migration_v1.md": (
            "70f917cd6ae0123d99d7166d08bd0cbac91d49d88fb2d8b830e7e0bbec2d0a7a"
        ),
        "examples/promptsec-annotation-example-v1.json": (
            "f6157c10a34dbb57cfca2aeffce42e1cdf1df97a204e6aa0c109f983d61dba6e"
        ),
        "schemas/promptsec-annotation-v1.schema.json": (
            "9dd37ad47f052daff2deb4e11639b1096f43214fb9804991c95bf08d770819b0"
        ),
    }

    actual = {
        relative_path: hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in expected_sha256
    }
    assert actual == expected_sha256
