"""Versioned prompt loading and safe, deterministic template rendering."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptsec.data.hashing import sha256_text

_SAFE_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_VERSION = re.compile(r"^v[1-9][0-9]*$")
_DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parents[3] / "prompts" / "policybench"


class PromptError(ValueError):
    """Raised when a prompt file or interpolation request is unsafe."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    text: str
    sha256: str
    path: Path

    @property
    def prompt_version(self) -> str:
        return f"{self.name}_{self.version}"

    @property
    def fields(self) -> tuple[str, ...]:
        fields: list[str] = []
        for _, field_name, format_spec, conversion in string.Formatter().parse(self.text):
            if field_name is None:
                continue
            if (
                not field_name.isidentifier()
                or "." in field_name
                or "[" in field_name
                or conversion
                or format_spec
            ):
                raise PromptError(
                    f"{self.path}: prompt placeholders must be plain identifiers without "
                    "conversions or format specifications"
                )
            if field_name not in fields:
                fields.append(field_name)
        return tuple(fields)

    def render(self, **values: Any) -> str:
        """Render only explicitly declared placeholders with no attribute traversal."""

        expected = set(self.fields)
        supplied = set(values)
        if expected != supplied:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise PromptError(
                f"prompt values do not match {self.prompt_version}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        rendered_values: dict[str, str] = {}
        for key, value in values.items():
            if isinstance(value, str):
                rendered_values[key] = value
            else:
                rendered_values[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        return self.text.format_map(rendered_values)


class PromptRepository:
    """Load prompt files confined to one explicit directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_PROMPT_ROOT

    def load(self, name: str, version: str = "v1") -> PromptTemplate:
        if not isinstance(name, str) or not _SAFE_COMPONENT.fullmatch(name):
            raise PromptError(f"unsafe prompt name: {name!r}")
        if not isinstance(version, str) or not _SAFE_VERSION.fullmatch(version):
            raise PromptError(f"unsafe prompt version: {version!r}")
        return self.load_filename(f"{name}_{version}.txt")

    def load_filename(self, filename: str) -> PromptTemplate:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise PromptError(f"unsafe prompt filename: {filename!r}")
        match = re.fullmatch(r"([a-z][a-z0-9_]*)_(v[1-9][0-9]*)\.txt", filename)
        if match is None:
            raise PromptError(f"prompt filename must use NAME_vN.txt: {filename!r}")
        root = self.root.resolve()
        path = (root / filename).resolve()
        if path.parent != root:
            raise PromptError(f"prompt path escapes configured root: {filename!r}")
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"PolicyBench prompt not found: {path}") from error
        except (OSError, UnicodeError) as error:
            raise PromptError(f"cannot read PolicyBench prompt {path}: {error}") from error
        if not text.strip():
            raise PromptError(f"PolicyBench prompt is empty: {path}")
        template = PromptTemplate(
            name=match.group(1),
            version=match.group(2),
            text=text,
            sha256=sha256_text(text),
            path=path,
        )
        # Validate placeholder safety immediately, rather than only when rendered.
        _ = template.fields
        return template


@dataclass(frozen=True, slots=True)
class PromptBundle:
    system: PromptTemplate
    generate_policy_variant: PromptTemplate
    generate_scenario: PromptTemplate
    paraphrase: PromptTemplate
    validate_semantics: PromptTemplate
    codex_batch: PromptTemplate

    @classmethod
    def load(cls, root: str | Path | None = None) -> PromptBundle:
        repository = PromptRepository(root)
        return cls(
            system=repository.load("system"),
            generate_policy_variant=repository.load("generate_policy_variant"),
            generate_scenario=repository.load("generate_scenario"),
            paraphrase=repository.load("paraphrase"),
            validate_semantics=repository.load("validate_semantics"),
            codex_batch=repository.load("codex_batch", "v9"),
        )

    def hashes(self) -> dict[str, str]:
        return {
            prompt.prompt_version: prompt.sha256
            for prompt in (
                self.system,
                self.generate_policy_variant,
                self.generate_scenario,
                self.paraphrase,
                self.validate_semantics,
            )
        }


__all__ = [
    "PromptBundle",
    "PromptError",
    "PromptRepository",
    "PromptTemplate",
]
