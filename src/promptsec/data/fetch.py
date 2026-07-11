"""Download configured immutable artifacts and verify their checksums."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from promptsec.data.config import SourceConfig
from promptsec.data.hashing import sha256_file


class FetchError(RuntimeError):
    """Raised when an upstream artifact cannot be acquired safely."""


def _filename(url: str, artifact_id: str) -> str:
    name = PurePosixPath(urlparse(url).path).name
    return name or artifact_id


def fetch_artifacts(
    config: SourceConfig,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    root = Path(destination) / config.id / (config.revision or config.version or "unversioned")
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for artifact in config.artifacts:
        target = root / _filename(artifact.url, artifact.id)
        if target.exists() and not overwrite:
            actual = sha256_file(target)
            if artifact.sha256 and actual != artifact.sha256:
                raise FetchError(
                    f"existing {target} has checksum {actual}, expected {artifact.sha256}"
                )
            outputs[artifact.id] = target
            continue

        temporary = target.with_name(f".{target.name}.tmp")
        request = Request(artifact.url, headers={"User-Agent": "PromptSec-Dataset/0.1"})
        try:
            with urlopen(request) as response, temporary.open("wb") as stream:  # noqa: S310
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            actual = sha256_file(temporary)
            if artifact.sha256 and actual != artifact.sha256:
                raise FetchError(
                    f"downloaded {artifact.id} has checksum {actual}, expected {artifact.sha256}"
                )
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        outputs[artifact.id] = target
    return outputs
