"""Download configured immutable artifacts and verify their checksums."""

from __future__ import annotations

import os
import subprocess
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
    offline: bool = False,
    local_path: str | Path | None = None,
) -> dict[str, Path]:
    if local_path is not None:
        return _local_artifacts(config, Path(local_path))

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

        if offline:
            raise FetchError(f"offline mode: missing pinned artifact {artifact.id!r} at {target}")

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


def _local_artifacts(config: SourceConfig, local_path: Path) -> dict[str, Path]:
    candidate = local_path.resolve()
    if not candidate.exists():
        raise FetchError(f"local source override does not exist: {candidate}")
    if candidate.is_file():
        if len(config.artifacts) != 1:
            raise FetchError(
                f"source {config.id} has {len(config.artifacts)} artifacts; "
                "a file override is ambiguous"
            )
        artifact = config.artifacts[0]
        _require_checksum(candidate, artifact.id, artifact.sha256)
        return {artifact.id: candidate}

    revision_verified = _verify_git_revision(candidate, config.revision)
    if revision_verified:
        _require_clean_checkout(candidate)
    outputs: dict[str, Path] = {}
    for artifact in config.artifacts:
        if artifact.local_path:
            resolved = (candidate / artifact.local_path).resolve()
            if not resolved.is_relative_to(candidate):
                raise FetchError(f"artifact {artifact.id!r} local_path escapes override root")
        elif len(config.artifacts) == 1 and revision_verified:
            resolved = candidate
        else:
            raise FetchError(f"artifact {artifact.id!r} has no local_path for directory override")
        if not resolved.exists():
            raise FetchError(f"local artifact {artifact.id!r} is missing: {resolved}")
        if resolved.is_file() and not revision_verified:
            _require_checksum(resolved, artifact.id, artifact.sha256)
        elif not revision_verified:
            raise FetchError(
                f"cannot verify directory artifact {artifact.id!r} without a matching Git revision"
            )
        outputs[artifact.id] = resolved
    return outputs


def _require_checksum(path: Path, artifact_id: str, expected: str | None) -> None:
    if not path.is_file():
        raise FetchError(f"local artifact {artifact_id!r} is not a file: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise FetchError(
            f"local artifact {artifact_id!r} has checksum {actual}, expected {expected}"
        )


def _verify_git_revision(path: Path, expected: str | None) -> bool:
    if not (path / ".git").exists():
        return False
    if not expected:
        raise FetchError(f"local Git checkout has no configured revision: {path}")
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise FetchError(f"cannot resolve local Git revision for {path}: {exc}") from exc
    actual = result.stdout.strip().lower()
    if actual != expected.lower():
        raise FetchError(
            f"local checkout revision {actual} does not match pinned revision {expected}"
        )
    return True


def _require_clean_checkout(path: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise FetchError(f"cannot verify local Git checkout cleanliness for {path}: {exc}") from exc
    if result.stdout.strip():
        raise FetchError(f"local Git checkout has tracked modifications: {path}")
