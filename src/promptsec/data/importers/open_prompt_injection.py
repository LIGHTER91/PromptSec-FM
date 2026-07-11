"""Importer for Open-Prompt-Injection's allow-listed benchmark definitions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from promptsec.data.importers.base import (
    BaseImporter,
    ImporterError,
    RawRecord,
    portable_path,
)

_CONFIG_DIRECTORY = ("configs", "task_configs")
_TEMPLATE_DIRECTORY = ("data", "system_prompts")
_INJECTED_VARIANTS = (
    ("", "base"),
    ("_short", "short"),
    ("_med_long", "med_long"),
    ("_long", "long"),
)
_MAX_MEMBER_BYTES = 1024 * 1024
_REVIEW_REASON = (
    "Only Open-Prompt-Injection benchmark task definitions and instruction templates are "
    "imported; downstream dataset examples are not redistributed and require source-specific "
    "review."
)


class OpenPromptInjectionImporter(BaseImporter):
    """Resolve task/injection templates directly from a pinned ZIP archive."""

    def source_record_id(self, raw: RawRecord) -> str:
        task_config = raw.payload.get("task_config")
        template_name = raw.payload.get("template_name")
        if isinstance(task_config, str) and isinstance(template_name, str):
            return f"{task_config}::{template_name}"
        return super().source_record_id(raw)

    def iter_raw_records(self, inputs: Mapping[str, str | Path]) -> Iterator[RawRecord]:
        resolved_inputs = self._validated_inputs(inputs)
        emitted_index = 0
        for artifact in self.config.artifacts:
            if artifact.format != "zip":
                raise ImporterError(
                    "Open-Prompt-Injection artifacts must use the zip format; "
                    f"got {artifact.format!r} for {artifact.id!r}"
                )
            archive_path = resolved_inputs[artifact.id]
            try:
                with ZipFile(archive_path, "r") as archive:
                    members = self._allowlisted_members(archive, archive_path)
                    config_names = sorted(
                        name for name in members if name.startswith("configs/task_configs/")
                    )
                    if not config_names:
                        raise ImporterError(
                            f"{archive_path}: no allow-listed configs/task_configs/*.json "
                            "members found"
                        )
                    for config_name in config_names:
                        config = self._read_config(
                            archive,
                            members[config_name],
                            archive_path=archive_path,
                            encoding=artifact.encoding,
                        )
                        task_info = config.get("task_info")
                        if task_info is None:
                            continue
                        if not isinstance(task_info, dict):
                            raise ImporterError(
                                f"{archive_path}:{config_name}: task_info must be an object"
                            )
                        target_reference = task_info.get("target_instruction")
                        injected_reference = task_info.get("injected_instruction")
                        if not (
                            isinstance(target_reference, str)
                            and target_reference
                            and isinstance(injected_reference, str)
                            and injected_reference
                        ):
                            continue

                        dataset_info = config.get("dataset_info", {})
                        if not isinstance(dataset_info, dict):
                            raise ImporterError(
                                f"{archive_path}:{config_name}: dataset_info must be an object"
                            )
                        target_stem = self._template_stem(
                            target_reference,
                            config_name=config_name,
                            field="target_instruction",
                        )
                        injected_stem = self._template_stem(
                            injected_reference,
                            config_name=config_name,
                            field="injected_instruction",
                        )
                        target_template = f"data/system_prompts/{target_stem}.txt"
                        target_info = self._required_template(
                            members,
                            target_template,
                            archive_path=archive_path,
                            config_name=config_name,
                        )
                        target_text = self._read_text_member(
                            archive,
                            target_info,
                            archive_path=archive_path,
                            encoding=artifact.encoding,
                        )

                        base_template = f"data/system_prompts/{injected_stem}.txt"
                        self._required_template(
                            members,
                            base_template,
                            archive_path=archive_path,
                            config_name=config_name,
                        )
                        for suffix, variant in _INJECTED_VARIANTS:
                            template_name = f"{injected_stem}{suffix}"
                            template_path = f"data/system_prompts/{template_name}.txt"
                            template_info = members.get(template_path)
                            if template_info is None:
                                # Only the configured base reference is mandatory.  The
                                # length variants are imported whenever upstream provides them.
                                continue
                            injected_text = self._read_text_member(
                                archive,
                                template_info,
                                archive_path=archive_path,
                                encoding=artifact.encoding,
                            )
                            payload = {
                                "task_config": config_name,
                                "config": config,
                                "task": task_info.get("task"),
                                "task_type": task_info.get("type"),
                                "dataset": dataset_info.get("dataset"),
                                "target_instruction_name": target_reference,
                                "injected_instruction_name": injected_reference,
                                "variant": variant,
                                "template_name": template_name,
                                "target_template_path": target_template,
                                "injected_template_path": template_path,
                                "target_instruction": target_text,
                                "injected_instruction": injected_text,
                            }
                            yield RawRecord(
                                artifact=artifact,
                                raw_artifact=portable_path(archive_path),
                                split=artifact.split,
                                index=emitted_index,
                                payload=payload,
                            )
                            emitted_index += 1
            except (BadZipFile, OSError) as exc:
                raise ImporterError(
                    f"cannot read Open-Prompt-Injection archive {archive_path}: {exc}"
                ) from exc

    def transform(self, raw: RawRecord) -> Iterable[dict[str, Any]]:
        injected_instruction = raw.payload.get("injected_instruction")
        target_instruction = raw.payload.get("target_instruction")
        if not isinstance(injected_instruction, str):
            raise ImporterError("resolved injected_instruction must be a string")
        if not isinstance(target_instruction, str):
            raise ImporterError("resolved target_instruction must be a string")

        annotations = {
            "instruction_presence": "INSTRUCTION_PRESENT",
            "instruction_presentation": "OPERATIVE",
            "instruction_addressee": "MODEL_OR_AGENT",
            "user_goal_alignment": "MISALIGNED",
            "protected_policy_alignment": "UNDETERMINED",
            "authority_status": "OUTSIDE_AUTHORITY",
            "attack_families": ["PROMPT_INJECTION"],
            "attack_objectives": ["TASK_HIJACKING"],
            "spans": [
                {
                    "start": 0,
                    "end": len(injected_instruction),
                    "type": "INJECTION_PAYLOAD",
                }
            ],
            "annotation_status": "CONFIRMED",
            "annotator_confidence": 1.0,
        }
        original_label_fields = (
            "task",
            "task_type",
            "dataset",
            "target_instruction_name",
            "injected_instruction_name",
            "variant",
            "template_name",
        )
        original_labels = {field: raw.payload.get(field) for field in original_label_fields}
        yield self.build_record(
            raw,
            text=injected_instruction,
            source_text=injected_instruction,
            annotations=annotations,
            original_labels=original_labels,
            mapping_status="NEEDS_REVIEW",
            field_mappings=[
                {
                    "source": "injected_instruction",
                    "target": "content.text",
                    "method": "COPY_EXACT_RESOLVED_TEMPLATE",
                },
                {
                    "source": "target_instruction",
                    "target": "context.user_goal",
                    "method": "COPY_EXACT_RESOLVED_TEMPLATE",
                },
                {
                    "source": "task_config",
                    "target": "annotations.attack_families",
                    "method": "EXPLICIT_TARGET_INJECTED_PAIR_TO_PROMPT_INJECTION",
                },
                {
                    "source": "target_instruction_name,injected_instruction_name",
                    "target": "annotations.attack_objectives",
                    "method": "EXPLICIT_REPLACEMENT_TASK_PAIR_TO_TASK_HIJACKING",
                },
            ],
            review_reasons=[_REVIEW_REASON],
            transformations=["RESOLVE_ALLOWLISTED_INSTRUCTION_TEMPLATES"],
            context={"user_goal": target_instruction},
            content_overrides={
                "delivery_mode": "INDIRECT",
                "source_role": "EXTERNAL_CONTENT",
            },
        )

    def _validated_inputs(self, inputs: Mapping[str, str | Path]) -> dict[str, Path]:
        expected = {artifact.id for artifact in self.config.artifacts}
        unexpected = set(inputs) - expected
        if unexpected:
            raise ImporterError(f"unknown input artifact(s): {sorted(unexpected)}")
        missing = expected - set(inputs)
        if missing:
            raise ImporterError(f"missing input artifact(s): {sorted(missing)}")
        return {artifact_id: Path(path) for artifact_id, path in inputs.items()}

    @staticmethod
    def _allowlisted_members(
        archive: ZipFile,
        archive_path: Path,
    ) -> dict[str, ZipInfo]:
        members: dict[str, ZipInfo] = {}
        for info in archive.infolist():
            relative_name = OpenPromptInjectionImporter._allowlisted_name(info.filename)
            if relative_name is None or info.is_dir():
                continue
            if relative_name in members:
                raise ImporterError(
                    f"{archive_path}: duplicate allow-listed ZIP member {relative_name!r}"
                )
            members[relative_name] = info
        return members

    @staticmethod
    def _allowlisted_name(name: str) -> str | None:
        path = PurePosixPath(name)
        parts = path.parts
        if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
            return None
        if len(parts) < 3:
            return None
        directory = tuple(parts[-3:-1])
        filename = parts[-1]
        if directory == _CONFIG_DIRECTORY and filename.endswith(".json"):
            return "/".join(parts[-3:])
        if directory == _TEMPLATE_DIRECTORY and filename.endswith(".txt"):
            return "/".join(parts[-3:])
        return None

    @staticmethod
    def _read_config(
        archive: ZipFile,
        info: ZipInfo,
        *,
        archive_path: Path,
        encoding: str,
    ) -> dict[str, Any]:
        text = OpenPromptInjectionImporter._read_text_member(
            archive,
            info,
            archive_path=archive_path,
            encoding=encoding,
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImporterError(
                f"{archive_path}:{info.filename}: invalid task config JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ImporterError(f"{archive_path}:{info.filename}: task config must be an object")
        return value

    @staticmethod
    def _read_text_member(
        archive: ZipFile,
        info: ZipInfo,
        *,
        archive_path: Path,
        encoding: str,
    ) -> str:
        if info.file_size > _MAX_MEMBER_BYTES:
            raise ImporterError(
                f"{archive_path}:{info.filename}: allow-listed member exceeds "
                f"{_MAX_MEMBER_BYTES} bytes"
            )
        try:
            with archive.open(info, "r") as stream:
                data = stream.read(_MAX_MEMBER_BYTES + 1)
            if len(data) > _MAX_MEMBER_BYTES:
                raise ImporterError(
                    f"{archive_path}:{info.filename}: allow-listed member exceeds "
                    f"{_MAX_MEMBER_BYTES} bytes"
                )
            return data.decode(encoding)
        except (BadZipFile, OSError, RuntimeError, UnicodeError) as exc:
            raise ImporterError(
                f"cannot read allow-listed ZIP member {archive_path}:{info.filename}: {exc}"
            ) from exc

    @staticmethod
    def _template_stem(reference: str, *, config_name: str, field: str) -> str:
        stem = reference[:-4] if reference.endswith(".txt") else reference
        if not stem or "\\" in stem or PurePosixPath(stem).name != stem or stem in {".", ".."}:
            raise ImporterError(
                f"{config_name}: {field} contains an unsafe template reference {reference!r}"
            )
        return stem

    @staticmethod
    def _required_template(
        members: Mapping[str, ZipInfo],
        template_path: str,
        *,
        archive_path: Path,
        config_name: str,
    ) -> ZipInfo:
        info = members.get(template_path)
        if info is None:
            raise ImporterError(
                f"{archive_path}:{config_name}: referenced template is missing: {template_path}"
            )
        return info


__all__ = ["OpenPromptInjectionImporter"]
