#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the deterministic GitHub-first PromptSec-FM Colab notebook."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "notebooks" / "PromptSec_FM_XLMR_Multitask_Colab.ipynb"

SECTION_TITLES = (
    "1. Titre et objectif scientifique",
    "2. Avertissement sur les données SILVER",
    "3. Configuration utilisateur",
    "4. Préflight du runtime Colab et du GPU",
    "5. Montage de Google Drive",
    "6. Clonage ou mise à jour du dépôt GitHub",
    "7. Vérification du commit du dépôt",
    "8. Installation des dépendances et du projet éditable",
    "9. Vérification de la présence de l’archive",
    "10. Vérification SHA-256",
    "11. Extraction locale sécurisée",
    "12. Validation du dataset et des splits officiels",
    "13. Inspection des vocabulaires d’étiquettes",
    "14. Aperçu sérialisé expurgé",
    "15. Inventaire des checkpoints",
    "16. Smoke test optionnel",
    "17. Entraînement complet d’évaluation scientifique",
    "18. Reprise automatique",
    "19. Évaluation du meilleur checkpoint",
    "20. Analyse contrefactuelle",
    "21. Analyse des négatifs difficiles",
    "22. Analyse anglais/français",
    "23. Inventaire du modèle final et des checkpoints",
    "24. Aperçu des rapports",
    "25. Résumé de reproductibilité",
    "26. Conseils de nettoyage",
)


def _clean(value: str) -> str:
    return textwrap.dedent(value).strip()


def _markdown(title: str, content: str) -> Any:
    return new_markdown_cell(f"## {title}\n\n{_clean(content)}")


def _code(content: str) -> Any:
    return new_code_cell(_clean(content))


def build_notebook() -> Any:
    cells = [
        _markdown(
            SECTION_TITLES[0],
            """
            Ce notebook orchestre l’entraînement multilingue et multi-tâche de
            `FacebookAI/xlm-roberta-base` sur le contexte PromptSec-FM complet. Le dépôt source
            est toujours cloné depuis GitHub dans le runtime Colab éphémère. Google Drive ne
            conserve que l’archive de données, les checkpoints, les rapports et le modèle exporté.

            Les sept têtes mono-étiquette et les deux têtes multi-étiquettes sont implémentées par
            le CLI testé du dépôt. Aucune boucle d’entraînement n’est dupliquée ici.
            """,
        ),
        _markdown(
            SECTION_TITLES[1],
            """
            **Les 6 000 annotations sont synthétiques et `SILVER_VALIDATED`, pas des annotations
            humaines Gold.** Les résultats ne constituent pas une évaluation human-Gold. Une
            bonne performance sur PolicyBench ne prouve pas une robustesse en conditions réelles.

            Le mode scientifique entraîne uniquement sur le split officiel `train`; la validation
            sélectionne le checkpoint. Les tests OOD et contrefactuels ne servent jamais à la
            sélection. Le fichier `human_review_candidates.jsonl` n’est jamais entraîné.
            """,
        ),
        _markdown(
            SECTION_TITLES[2],
            """
            Modifier cette cellule avant **Runtime → Run all**. Pour une reproductibilité stricte,
            remplacer `GITHUB_REF` par un tag de release ou un SHA de commit exact. Le dépôt public
            fonctionne par défaut, sans URL privée, identifiant Drive privé ni credential.
            """,
        ),
        _code(
            """
            GITHUB_OWNER = "LIGHTER91"
            GITHUB_REPOSITORY = "PromptSec-FM"
            GITHUB_REF = "main"

            REPO_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}.git"
            REPO_DIR = "/content/PromptSec-FM"

            DRIVE_ROOT = "/content/drive/MyDrive/PromptSec-FM"
            DATA_ARCHIVE = f"{DRIVE_ROOT}/data/policybench-codex-v0.1.zip"
            DATA_SHA256_FILE = DATA_ARCHIVE + ".sha256"
            LOCAL_DATA_ROOT = "/content/promptsec_data"
            DATASET_DIR = f"{LOCAL_DATA_ROOT}/policybench-codex-v0.1"

            RUN_NAME = "xlmr-base-multitask-v0.1"
            CHECKPOINT_ROOT = f"{DRIVE_ROOT}/checkpoints/{RUN_NAME}"
            REPORT_ROOT = f"{DRIVE_ROOT}/reports/{RUN_NAME}"

            TRAINING_MODE = "SCIENTIFIC_EVALUATION"
            MODEL_NAME = "FacebookAI/xlm-roberta-base"
            MAX_LENGTH = 512
            NUM_EPOCHS = 4
            LEARNING_RATE = 2e-5
            WEIGHT_DECAY = 0.01
            WARMUP_RATIO = 0.10
            EARLY_STOPPING_PATIENCE = 2
            SEED = 20260718

            RESUME = True
            RUN_SMOKE_TEST_FIRST = True
            START_FULL_TRAINING = False
            ALLOW_CPU_SMOKE_TEST = False
            FORCE_RECLONE = False
            INSTALL_DEV_DEPENDENCIES = False
            RUN_INFERENCE_EXAMPLE = False
            CLEAN_LOCAL_RUNTIME = False

            REFERENCE_ARCHIVE_SHA256 = (
                "0ccb70cb1db7c38ad52ba0c395a5b2a62a72d57870ca1af86711037c2e9a59e5"
            )
            """
        ),
        _markdown(
            SECTION_TITLES[3],
            """
            Le préflight décrit le runtime avant toute installation. L’entraînement complet exige
            CUDA. Un smoke test CPU n’est autorisé que si `ALLOW_CPU_SMOKE_TEST=True` et ne donne
            aucune indication représentative de la vitesse d’entraînement.
            """,
        ),
        _code(
            """
            import importlib.metadata
            import os
            import platform
            import shutil
            from pathlib import Path

            import torch

            print("Préflight du runtime Colab")
            print("Python:", platform.python_version())
            print("Système:", platform.platform())
            print("PyTorch:", torch.__version__)
            try:
                print("Transformers:", importlib.metadata.version("transformers"))
            except importlib.metadata.PackageNotFoundError:
                print("Transformers: pas encore installé")

            CUDA_AVAILABLE = torch.cuda.is_available()
            print("CUDA disponible:", CUDA_AVAILABLE)
            print("Runtime CUDA:", torch.version.cuda)
            if CUDA_AVAILABLE:
                free_vram, total_vram = torch.cuda.mem_get_info()
                TOTAL_VRAM_GIB = total_vram / 2**30
                FREE_VRAM_GIB = free_vram / 2**30
                BF16_SUPPORTED = bool(torch.cuda.is_bf16_supported())
                GPU_NAME = torch.cuda.get_device_name(0)
            else:
                TOTAL_VRAM_GIB = 0.0
                FREE_VRAM_GIB = 0.0
                BF16_SUPPORTED = False
                GPU_NAME = None

            if TOTAL_VRAM_GIB >= 15:
                PER_DEVICE_BATCH_SIZE = 8
                GRADIENT_ACCUMULATION_STEPS = 2
            elif TOTAL_VRAM_GIB >= 10:
                PER_DEVICE_BATCH_SIZE = 4
                GRADIENT_ACCUMULATION_STEPS = 4
            else:
                PER_DEVICE_BATCH_SIZE = 2
                GRADIENT_ACCUMULATION_STEPS = 8
            EFFECTIVE_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            PRECISION_MODE = "bf16" if BF16_SUPPORTED else ("fp16" if CUDA_AVAILABLE else "fp32")

            print("GPU:", GPU_NAME or "aucun")
            print("VRAM totale/libre (GiB):", TOTAL_VRAM_GIB, FREE_VRAM_GIB)
            print("bf16 supporté:", BF16_SUPPORTED)
            print("Précision résolue:", PRECISION_MODE)
            print("Batch par device:", PER_DEVICE_BATCH_SIZE)
            print("Accumulation de gradients:", GRADIENT_ACCUMULATION_STEPS)
            print("Batch effectif:", EFFECTIVE_BATCH_SIZE)

            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            print(
                "RAM système totale/disponible (GiB):",
                page_size * total_pages / 2**30,
                page_size * available_pages / 2**30,
            )
            if Path(DRIVE_ROOT).exists():
                print("Espace Drive disponible (GiB):", shutil.disk_usage(DRIVE_ROOT).free / 2**30)
            else:
                print("Espace Drive: mesuré après montage")
            if not CUDA_AVAILABLE:
                print("L’entraînement complet restera bloqué tant qu’un runtime GPU n’est pas sélectionné.")
            """
        ),
        _markdown(
            SECTION_TITLES[4],
            """
            Monter Drive puis créer uniquement `data/`, le répertoire de checkpoints et celui des
            rapports. Le dépôt GitHub n’est jamais copié dans Drive et l’entraînement ne lit jamais
            directement le ZIP.
            """,
        ),
        _code(
            """
            import shutil
            from pathlib import Path

            from google.colab import drive

            print("Montage de Google Drive")
            drive.mount("/content/drive")
            for persistent_directory in (
                Path(DRIVE_ROOT) / "data",
                Path(CHECKPOINT_ROOT),
                Path(REPORT_ROOT),
            ):
                persistent_directory.mkdir(parents=True, exist_ok=True)
            print("Drive root:", DRIVE_ROOT)
            print("Espace Drive disponible (GiB):", shutil.disk_usage(DRIVE_ROOT).free / 2**30)
            """
        ),
        _markdown(
            SECTION_TITLES[5],
            """
            Le dépôt complet est cloné sous `/content`. Une réexécution vérifie le remote, refuse
            un arbre de travail sale, fetch `origin`, puis effectue une mise à jour fast-forward
            pour une branche ou un checkout détaché pour un tag/SHA. `FORCE_RECLONE` ne peut
            supprimer que `REPO_DIR` sous `/content`, jamais Drive.
            """,
        ),
        _code(
            """
            import os
            import shutil
            import subprocess  # noqa: F811
            from pathlib import Path

            print("Préparation du dépôt GitHub:", REPO_URL, "ref:", GITHUB_REF)
            repo_path = Path(REPO_DIR).resolve()
            content_root = Path("/content").resolve()
            drive_mount = Path("/content/drive").resolve()
            if content_root not in repo_path.parents or drive_mount in repo_path.parents:
                raise RuntimeError("REPO_DIR doit rester sous /content et hors de Google Drive")

            def run_git(arguments, *, cwd=None, capture=False, check=True):
                return subprocess.run(
                    ["git", *arguments],
                    cwd=cwd,
                    check=check,
                    shell=False,
                    text=True,
                    capture_output=capture,
                )

            def git_output(arguments, *, cwd):
                return run_git(arguments, cwd=cwd, capture=True).stdout.strip()

            def normalized_remote(value):
                return value.rstrip("/").removesuffix(".git").lower()

            if FORCE_RECLONE and repo_path.exists():
                print("FORCE_RECLONE=True: suppression du seul dépôt éphémère", repo_path)
                shutil.rmtree(repo_path)

            branch_query = run_git(
                ["ls-remote", "--heads", REPO_URL, GITHUB_REF], capture=True
            ).stdout.strip()
            tag_query = run_git(
                ["ls-remote", "--tags", REPO_URL, f"refs/tags/{GITHUB_REF}"], capture=True
            ).stdout.strip()

            if not repo_path.exists():
                if branch_query or tag_query:
                    run_git(
                        [
                            "clone",
                            "--branch",
                            GITHUB_REF,
                            "--single-branch",
                            REPO_URL,
                            str(repo_path),
                        ]
                    )
                else:
                    run_git(["clone", "--no-checkout", REPO_URL, str(repo_path)])
                    run_git(["checkout", "--detach", GITHUB_REF], cwd=repo_path)
            else:
                if not (repo_path / ".git").is_dir():
                    raise RuntimeError(
                        "REPO_DIR existe sans être un dépôt Git; utiliser un autre chemin ou FORCE_RECLONE=True"
                    )
                existing_remote = git_output(["remote", "get-url", "origin"], cwd=repo_path)
                if normalized_remote(existing_remote) != normalized_remote(REPO_URL):
                    raise RuntimeError(
                        f"Remote inattendu: {existing_remote}. Utiliser FORCE_RECLONE=True si approprié."
                    )
                dirty = git_output(["status", "--porcelain"], cwd=repo_path)
                if dirty:
                    raise RuntimeError(
                        "Le dépôt Colab contient des changements locaux. Ils ne seront pas détruits; "
                        "inspecter l’état ou utiliser FORCE_RECLONE=True."
                    )
                run_git(["fetch", "--tags", "--prune", "origin"], cwd=repo_path)
                if branch_query:
                    run_git(
                        [
                            "fetch",
                            "origin",
                            f"refs/heads/{GITHUB_REF}:refs/remotes/origin/{GITHUB_REF}",
                        ],
                        cwd=repo_path,
                    )
                    local_branch = run_git(
                        ["show-ref", "--verify", f"refs/heads/{GITHUB_REF}"],
                        cwd=repo_path,
                        check=False,
                    ).returncode == 0
                    if local_branch:
                        run_git(["checkout", GITHUB_REF], cwd=repo_path)
                    else:
                        run_git(
                            ["checkout", "-b", GITHUB_REF, "--track", f"origin/{GITHUB_REF}"],
                            cwd=repo_path,
                        )
                    run_git(["merge", "--ff-only", f"origin/{GITHUB_REF}"], cwd=repo_path)
                else:
                    if tag_query:
                        run_git(["fetch", "origin", f"refs/tags/{GITHUB_REF}"], cwd=repo_path)
                    else:
                        run_git(["fetch", "origin", GITHUB_REF], cwd=repo_path)
                    run_git(["checkout", "--detach", GITHUB_REF], cwd=repo_path)

            os.chdir(repo_path)
            print("Dépôt prêt:", Path.cwd())
            """
        ),
        _markdown(
            SECTION_TITLES[6],
            """
            Cette cellule affiche et conserve le SHA exact utilisé. Le CLI résout également le
            commit depuis son propre emplacement et l’inscrit dans le run manifest/checkpoint.
            """,
        ),
        _code(
            """
            import subprocess  # noqa: F811
            from pathlib import Path

            print("Vérification du commit source")

            def repository_value(arguments):
                return subprocess.run(
                    ["git", *arguments],
                    cwd=REPO_DIR,
                    check=True,
                    shell=False,
                    text=True,
                    capture_output=True,
                ).stdout.strip()

            REPOSITORY_COMMIT = repository_value(["rev-parse", "HEAD"])
            REPOSITORY_BRANCH = repository_value(["branch", "--show-current"]) or "DETACHED_HEAD"
            REPOSITORY_REMOTE = repository_value(["remote", "get-url", "origin"])
            LAST_COMMIT_TITLE = repository_value(["log", "-1", "--pretty=%s"])
            WORKTREE_STATUS = repository_value(["status", "--porcelain"])
            if WORKTREE_STATUS:
                raise RuntimeError("Le dépôt doit être propre avant installation et entraînement")
            print("Chemin:", Path(REPO_DIR).resolve())
            print("Branche/état:", REPOSITORY_BRANCH)
            print("Commit exact:", REPOSITORY_COMMIT)
            print("Remote:", REPOSITORY_REMOTE)
            print("Dernier commit:", LAST_COMMIT_TITLE)
            print("Arbre de travail: CLEAN")
            """
        ),
        _markdown(
            SECTION_TITLES[7],
            """
            Le groupe optionnel `training` défini dans `pyproject.toml` est installé en mode
            éditable. Aucune option `--upgrade` n’est utilisée: un PyTorch CUDA déjà compatible
            fourni par Colab est conservé. Activer les dépendances de développement uniquement
            pour exécuter les tests dans Colab.
            """,
        ),
        _code(
            """
            import importlib
            import importlib.metadata
            import subprocess  # noqa: F811
            import sys
            import tomllib
            from pathlib import Path

            print("Inspection de pyproject.toml et installation éditable")
            pyproject_path = Path(REPO_DIR) / "pyproject.toml"
            project_configuration = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            optional_groups = project_configuration.get("project", {}).get(
                "optional-dependencies", {}
            )
            print("Groupes optionnels:", sorted(optional_groups))
            if "training" not in optional_groups:
                raise RuntimeError("Le groupe optionnel training est absent de pyproject.toml")
            extras = ["training"]
            if INSTALL_DEV_DEPENDENCIES:
                extras.append("dev")
            editable_target = f".[{','.join(extras)}]"
            install_command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                editable_target,
            ]
            print("Commande d’installation:", install_command)
            subprocess.run(
                install_command,
                cwd=REPO_DIR,
                check=True,
                shell=False,
            )

            # Editable installs expose ``src`` through a .pth file, which is normally read only
            # when Python starts. The Colab kernel is already running because pip is invoked in a
            # subprocess, so make the freshly installed source tree importable immediately.
            expected_source = (Path(REPO_DIR) / "src").resolve()
            expected_source_text = str(expected_source)
            if expected_source_text not in sys.path:
                sys.path.insert(0, expected_source_text)
            importlib.invalidate_caches()

            distributions = {
                "promptsec": "promptsec-dataset",
                "torch": "torch",
                "transformers": "transformers",
                "tokenizers": "tokenizers",
                "safetensors": "safetensors",
                "scikit-learn": "scikit-learn",
                "numpy": "numpy",
                "pandas": "pandas",
                "pyyaml": "PyYAML",
            }
            for display_name, distribution_name in distributions.items():
                try:
                    version = importlib.metadata.version(distribution_name)
                except importlib.metadata.PackageNotFoundError:
                    version = "non installé (optionnel pour cet affichage)"
                print(f"{display_name}: {version}")

            training_package = importlib.import_module("promptsec.training")
            training_path = Path(training_package.__file__).resolve()
            if expected_source not in training_path.parents:
                raise RuntimeError(f"Import promptsec inattendu: {training_path}")
            print("Import promptsec.training vérifié depuis:", training_path)
            """
        ),
        _markdown(
            SECTION_TITLES[8],
            """
            Seuls le ZIP et son sidecar SHA-256 doivent être présents dans
            `MyDrive/PromptSec-FM/data/`. En cas d’absence, la cellule s’arrête avec les deux
            emplacements attendus.
            """,
        ),
        _code(
            """
            from pathlib import Path

            print("Vérification de la présence de l’archive Drive")
            archive_path = Path(DATA_ARCHIVE)
            sidecar_path = Path(DATA_SHA256_FILE)
            missing_paths = [path for path in (archive_path, sidecar_path) if not path.is_file()]
            if missing_paths:
                expected = "\\n".join(f"  - {path}" for path in (archive_path, sidecar_path))
                raise FileNotFoundError(
                    "Téléverser le ZIP et son sidecar dans Drive aux emplacements exacts:\\n"
                    + expected
                )
            print("Archive:", archive_path)
            print("Sidecar:", sidecar_path)
            """
        ),
        _markdown(
            SECTION_TITLES[9],
            """
            Le sidecar est la source de vérité opérationnelle. Le hash est calculé en streaming;
            la valeur de référence affichée dans la configuration n’est jamais utilisée à la place
            du sidecar. Une divergence bloque toute extraction.
            """,
        ),
        _code(
            """
            import hashlib  # noqa: F811
            import re
            from pathlib import Path

            print("Calcul SHA-256 en streaming")

            def sha256_stream(path, chunk_size=8 * 1024 * 1024):
                digest = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            sidecar_tokens = Path(DATA_SHA256_FILE).read_text(encoding="utf-8").split()
            if not sidecar_tokens or re.fullmatch(r"[0-9a-fA-F]{64}", sidecar_tokens[0]) is None:
                raise RuntimeError("Le sidecar SHA-256 est mal formé")
            EXPECTED_ARCHIVE_SHA256 = sidecar_tokens[0].lower()
            ARCHIVE_SHA256 = sha256_stream(DATA_ARCHIVE)
            print("Archive:", DATA_ARCHIVE)
            print("Taille (octets):", Path(DATA_ARCHIVE).stat().st_size)
            print("Hash attendu (sidecar):", EXPECTED_ARCHIVE_SHA256)
            print("Hash calculé:", ARCHIVE_SHA256)
            if ARCHIVE_SHA256 != EXPECTED_ARCHIVE_SHA256:
                print("SHA-256: FAIL")
                raise RuntimeError("Archive refusée: le SHA-256 ne correspond pas au sidecar")
            print("SHA-256: PASS")
            if ARCHIVE_SHA256 != REFERENCE_ARCHIVE_SHA256:
                print("Note: le sidecar valide une archive différente de la référence documentée.")
            """
        ),
        _markdown(
            SECTION_TITLES[10],
            """
            L’extraction reste sous `/content/promptsec_data`, refuse chemins absolus, traversées
            `..` et symlinks ZIP. Une extraction locale complète est réutilisée après vérification
            du manifeste embarqué; sinon seule la copie locale éphémère est supprimée et recréée.
            L’archive Drive n’est jamais supprimée.
            """,
        ),
        _code(
            """
            import hashlib  # noqa: F811
            import json  # noqa: F811
            import shutil
            import stat
            import zipfile
            from pathlib import Path, PurePosixPath

            print("Vérification ou extraction locale du dataset")
            local_root = Path(LOCAL_DATA_ROOT).resolve()
            dataset_path = Path(DATASET_DIR).resolve()
            content_root = Path("/content").resolve()
            if content_root not in local_root.parents or Path("/content/drive").resolve() in local_root.parents:
                raise RuntimeError("LOCAL_DATA_ROOT doit rester sous /content et hors de Drive")

            def file_sha256(path, chunk_size=8 * 1024 * 1024):
                digest = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def compatible_local_extraction(root):
                manifest_path = root / "colab_input_manifest.json"
                if not manifest_path.is_file():
                    return False
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    files = manifest["files"]
                    if manifest.get("release_id") != "policybench-codex-v0.1":
                        return False
                    for relative_name, expected_digest in files.items():
                        candidate = (root / relative_name).resolve()
                        if root not in candidate.parents or not candidate.is_file():
                            return False
                        if file_sha256(candidate) != expected_digest:
                            return False
                    return True
                except (OSError, ValueError, KeyError, TypeError):
                    return False

            if compatible_local_extraction(dataset_path):
                print("Extraction locale compatible déjà présente; réutilisation.")
            else:
                if local_root.exists():
                    print("Extraction locale absente/invalide; recréation sous", local_root)
                    shutil.rmtree(local_root)
                local_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(DATA_ARCHIVE) as archive:
                    for member in archive.infolist():
                        normalized_name = member.filename.replace("\\\\", "/")
                        relative = PurePosixPath(normalized_name)
                        unix_mode = member.external_attr >> 16
                        if relative.is_absolute() or ".." in relative.parts:
                            raise RuntimeError(f"Chemin ZIP dangereux refusé: {member.filename}")
                        if relative.parts and ":" in relative.parts[0]:
                            raise RuntimeError(f"Chemin ZIP absolu refusé: {member.filename}")
                        if stat.S_ISLNK(unix_mode):
                            raise RuntimeError(f"Symlink ZIP refusé: {member.filename}")
                        target = (local_root / Path(*relative.parts)).resolve()
                        if local_root not in target.parents and target != local_root:
                            raise RuntimeError(f"ZIP-slip refusé: {member.filename}")
                    archive.extractall(local_root)
                if not compatible_local_extraction(dataset_path):
                    raise RuntimeError("L’extraction locale ne passe pas les checksums embarqués")

            extracted_files = [path for path in dataset_path.rglob("*") if path.is_file()]
            extracted_bytes = sum(path.stat().st_size for path in extracted_files)
            print("Dataset local:", dataset_path)
            print("Fichiers extraits:", len(extracted_files))
            print("Taille locale (MiB):", extracted_bytes / 2**20)
            """
        ),
        _markdown(
            SECTION_TITLES[11],
            """
            Le loader du dépôt revalide schémas, checksums, identifiants, isolation des groupes,
            taxonomie et état SILVER avant tout chargement du modèle.

            Comptages officiels attendus: `train=1012`, `validation=242`,
            `test_policy_family_ood=284`, `test_domain_ood=491`,
            `test_language_ood=3000`, `test_counterfactual=344`,
            `human_review_candidates=627`. Aucun split aléatoire n’est créé.
            """,
        ),
        _code(
            """
            from promptsec.training.dataset import EXPECTED_SPLIT_COUNTS, load_training_dataset

            print("Validation complète des 6 000 records et des splits officiels")
            bundle = load_training_dataset(DATASET_DIR)
            integrity = bundle.integrity_report
            canonical = integrity["canonical_validation"]
            actual_split_counts = {
                split: len(records) for split, records in bundle.records_by_split.items()
            }
            if actual_split_counts != EXPECTED_SPLIT_COUNTS:
                raise RuntimeError(f"Comptages de splits inattendus: {actual_split_counts}")
            required_integrity = {
                "records": 6000,
                "unique_ids": 6000,
                "languages": {"en": 3000, "fr": 3000},
                "data_quality": "SILVER_VALIDATED",
                "human_validation_status": "PENDING",
                "annotator_confidence": 0.0,
                "automatic_gold_records": 0,
                "leakage_detected": False,
            }
            for field, expected_value in required_integrity.items():
                if integrity.get(field) != expected_value:
                    raise RuntimeError(
                        f"Intégrité invalide pour {field}: {integrity.get(field)!r}"
                    )
            if canonical.get("counterfactual_groups") != 720:
                raise RuntimeError("Le nombre de groupes contrefactuels doit être 720")
            if integrity.get("checksums_checked") != 17 or integrity.get("split_files_checked") != 7:
                raise RuntimeError("Les 17 checksums ou les sept fichiers de split ne sont pas validés")
            if bundle.split_audit.get("leakage_detected"):
                raise RuntimeError("Fuite détectée entre les splits officiels")
            print("Validation dataset: PASS")
            print("Records valides:", canonical["valid_records"])
            print("Langues:", integrity["languages"])
            print("Groupes contrefactuels:", canonical["counterfactual_groups"])
            print(
                "Checksums/splits vérifiés:",
                integrity["checksums_checked"],
                integrity["split_files_checked"],
            )
            print("Splits:", actual_split_counts)
            """
        ),
        _markdown(
            SECTION_TITLES[12],
            """
            Les neuf vocabulaires proviennent du schéma canonique gelé. Seuls dimensions, valeurs
            publiques de taxonomie et hashes de mapping sont affichés.
            """,
        ),
        _code(
            """
            print("Vocabulaires des neuf têtes")
            LABEL_DIMENSIONS = {}
            for head, mapping in bundle.mappings.items():
                LABEL_DIMENSIONS[head] = len(mapping.labels)
                print(
                    head,
                    "dimension=",
                    len(mapping.labels),
                    "multilabel=",
                    mapping.multilabel,
                    "labels=",
                    mapping.labels,
                    "hash=",
                    mapping.mapping_hash,
                )
            """
        ),
        _markdown(
            SECTION_TITLES[13],
            """
            L’aperçu ne révèle ni payload complet, ni blueprint caché, ni annotation automatique.
            Il affiche uniquement structure, longueurs, hash du texte candidat et dimensions des
            sorties.
            """,
        ),
        _code(
            """
            import hashlib  # noqa: F811

            from transformers import AutoTokenizer

            from promptsec.training.serialization import SPECIAL_TOKENS, record_sections
            from promptsec.training.token_budget import encode_with_section_budget

            print("Construction d’un aperçu structurel expurgé")
            preview_record = bundle.records_by_split["train"][0]
            preview_sections = record_sections(preview_record)
            preview_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            preview_tokenizer.add_special_tokens(
                {"additional_special_tokens": list(SPECIAL_TOKENS)}
            )
            preview_encoding = encode_with_section_budget(
                preview_tokenizer,
                preview_sections,
                max_length=MAX_LENGTH,
            )
            extension = preview_record["extensions"]["policybench_v0_1"]
            safe_preview = {
                "record_id": preview_record["id"],
                "language": preview_record["content"]["language"],
                "domain": extension["blueprint"]["domain"],
                "policy_present": preview_sections.protected_policy != "[NOT_PROVIDED]",
                "user_goal_present": preview_sections.user_goal != "[NOT_PROVIDED]",
                "source_role": preview_record["content"]["source_role"],
                "capability_count": len(
                    preview_record.get("context", {}).get("available_capabilities", [])
                ),
                "candidate_character_length": len(preview_sections.candidate),
                "candidate_sha256": hashlib.sha256(
                    preview_sections.candidate.encode("utf-8")
                ).hexdigest(),
                "serialized_token_length": len(preview_encoding.input_ids),
                "label_dimensions": LABEL_DIMENSIONS,
            }
            print(safe_preview)
            """
        ),
        _markdown(
            SECTION_TITLES[14],
            """
            Les manifests et checksums de tous les checkpoints sont vérifiés avant entraînement.
            Les checkpoints incomplets sont signalés; un checkpoint complet mais incompatible
            bloque la reprise. Les checkpoints smoke restent dans un sous-répertoire distinct.
            """,
        ),
        _code(
            """
            import dataclasses

            from promptsec.training.checkpoints import (
                IncompatibleCheckpointError,
                checkpoint_inventory,
                find_latest_compatible_checkpoint,
            )
            from promptsec.training.colab_config import load_training_config
            from promptsec.training.labels import mappings_fingerprint
            from promptsec.training.serialization import SPECIAL_TOKENS

            print("Inventaire et compatibilité des checkpoints complets")
            effective_settings = dataclasses.replace(
                load_training_config("configs/xlmr_multitask_colab_v0.1.yaml"),
                model_name=MODEL_NAME,
                training_mode=TRAINING_MODE,
                max_length=MAX_LENGTH,
                epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                warmup_ratio=WARMUP_RATIO,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                seed=SEED,
                per_device_batch_size=PER_DEVICE_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            )
            effective_settings.validate()
            expected_compatibility = {
                "dataset_manifest_sha256": bundle.manifest_sha256,
                "split_hashes": bundle.split_hashes,
                "training_config_hash": effective_settings.fingerprint(),
                "label_mapping_hash": mappings_fingerprint(bundle.mappings),
                "model_name": MODEL_NAME,
                "special_tokens": list(SPECIAL_TOKENS),
            }
            inventory_before_training = checkpoint_inventory(CHECKPOINT_ROOT)
            checkpoint_rows = []
            for item in inventory_before_training["checkpoints"]:
                manifest = item.get("manifest", {})
                state = manifest.get("trainer_state", {})
                checkpoint_rows.append(
                    {
                        "path": item["path"],
                        "status": item["status"],
                        "epoch": state.get("epoch"),
                        "global_step": state.get("global_step"),
                        "best_validation_score": state.get("best_validation_score"),
                        "training_config_hash": manifest.get("compatibility", {}).get(
                            "training_config_hash"
                        ),
                    }
                )
            print("Checkpoints détectés:", checkpoint_rows)
            try:
                LATEST_COMPATIBLE_CHECKPOINT = find_latest_compatible_checkpoint(
                    CHECKPOINT_ROOT,
                    compatibility=expected_compatibility,
                    run_kind=TRAINING_MODE,
                )
            except IncompatibleCheckpointError as error:
                raise RuntimeError(f"Checkpoint incompatible refusé: {error}") from error
            if LATEST_COMPATIBLE_CHECKPOINT is None:
                print("No compatible checkpoint found. Training will start from epoch zero.")
            else:
                print("Dernier checkpoint compatible:", LATEST_COMPATIBLE_CHECKPOINT)
            """
        ),
        _markdown(
            SECTION_TITLES[15],
            """
            Le smoke test appelle le CLI réel avec 32 records train, 16 validation, une époque et
            une longueur 128. Le CLI ajoute lui-même `smoke-test/` aux racines fournies. Une
            exécution smoke complète et vérifiée est réutilisée. Un état partiel déclenche la
            reprise sécurisée du CLI: seuls des checkpoints complets, checksummés et compatibles
            peuvent être restaurés; aucun artefact n’est supprimé automatiquement.
            """,
        ),
        _code(
            """
            import json  # noqa: F811
            import subprocess
            import sys
            from pathlib import Path

            from promptsec.training.checkpoints import checkpoint_inventory

            print("Préparation du smoke test")
            smoke_checkpoint_root = Path(CHECKPOINT_ROOT) / "smoke-test"
            smoke_report_root = Path(REPORT_ROOT) / "smoke-test"

            def completed_smoke_test():
                run_manifest_path = smoke_report_root / "run_manifest.json"
                probe_path = smoke_report_root / "smoke_resume_probe.json"
                if not run_manifest_path.is_file() or not probe_path.is_file():
                    return False
                run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                probe = json.loads(probe_path.read_text(encoding="utf-8"))
                complete = checkpoint_inventory(smoke_checkpoint_root)
                return (
                    run_manifest.get("status") == "COMPLETE"
                    and probe.get("status") == "PASS"
                    and any(item["status"] == "COMPLETE" for item in complete["checkpoints"])
                )

            SMOKE_TEST_PASSED = not RUN_SMOKE_TEST_FIRST
            if RUN_SMOKE_TEST_FIRST:
                if not CUDA_AVAILABLE and not ALLOW_CPU_SMOKE_TEST:
                    raise RuntimeError(
                        "Smoke CPU non autorisé. Sélectionner un GPU ou définir "
                        "ALLOW_CPU_SMOKE_TEST=True explicitement."
                    )
                if not CUDA_AVAILABLE:
                    print("Smoke test CPU explicite: la vitesse n’est pas représentative.")
                if completed_smoke_test():
                    print("Smoke test complet déjà présent; réutilisation des artefacts vérifiés.")
                    SMOKE_TEST_PASSED = True
                else:
                    smoke_resume_requested = (
                        smoke_checkpoint_root.exists() or smoke_report_root.exists()
                    )
                    if smoke_resume_requested:
                        smoke_inventory = checkpoint_inventory(smoke_checkpoint_root)
                        print(
                            "Artefacts smoke partiels détectés; tentative de reprise sécurisée.",
                            smoke_inventory,
                        )
                    smoke_command = [
                        sys.executable,
                        "scripts/train_xlmr_multitask.py",
                        "--config",
                        "configs/xlmr_multitask_colab_v0.1.yaml",
                        "--dataset",
                        DATASET_DIR,
                        "--output",
                        CHECKPOINT_ROOT,
                        "--reports",
                        REPORT_ROOT,
                        "--training-mode",
                        TRAINING_MODE,
                        "--model-name",
                        MODEL_NAME,
                        "--smoke-test",
                        "--max-train-records",
                        "32",
                        "--max-validation-records",
                        "16",
                        "--epochs",
                        "1",
                        "--max-length",
                        "128",
                        "--seed",
                        str(SEED),
                        "--per-device-batch-size",
                        str(PER_DEVICE_BATCH_SIZE),
                        "--gradient-accumulation-steps",
                        str(GRADIENT_ACCUMULATION_STEPS),
                        "--resume" if smoke_resume_requested else "--no-resume",
                    ]
                    print("Commande smoke:", smoke_command)
                    subprocess.run(smoke_command, cwd=REPO_DIR, check=True, shell=False)
                    if not completed_smoke_test():
                        raise RuntimeError("Le smoke test s’est terminé sans artefacts complets")
                    SMOKE_TEST_PASSED = True
            print("Smoke test validé:", SMOKE_TEST_PASSED)
            """
        ),
        _markdown(
            SECTION_TITLES[16],
            """
            `START_FULL_TRAINING=False` protège contre un lancement multi-heures accidentel. Ne le
            passer à `True` qu’après succès du smoke test. Le CLI conserve l’encodeur partagé, les
            neuf têtes, les splits officiels, la sélection validation et les analyses finales.
            """,
        ),
        _code(
            """
            import json  # noqa: F811
            import subprocess
            import sys
            from pathlib import Path

            print("Garde de l’entraînement scientifique complet")
            if not START_FULL_TRAINING:
                print(
                    "Entraînement complet désactivé. Après validation du smoke test, définir "
                    "START_FULL_TRAINING=True puis réexécuter cette cellule."
                )
            else:
                if not CUDA_AVAILABLE:
                    raise RuntimeError(
                        "CUDA est requis. Dans Colab: Runtime → Change runtime type → GPU."
                    )
                smoke_probe = Path(REPORT_ROOT) / "smoke-test" / "smoke_resume_probe.json"
                smoke_ok = not RUN_SMOKE_TEST_FIRST
                if smoke_probe.is_file():
                    smoke_ok = json.loads(smoke_probe.read_text(encoding="utf-8")).get(
                        "status"
                    ) == "PASS"
                if not smoke_ok:
                    raise RuntimeError("Le smoke test doit réussir avant l’entraînement complet")
                full_training_command = [
                    sys.executable,
                    "scripts/train_xlmr_multitask.py",
                    "--config",
                    "configs/xlmr_multitask_colab_v0.1.yaml",
                    "--dataset",
                    DATASET_DIR,
                    "--output",
                    CHECKPOINT_ROOT,
                    "--reports",
                    REPORT_ROOT,
                    "--training-mode",
                    TRAINING_MODE,
                    "--model-name",
                    MODEL_NAME,
                    "--learning-rate",
                    str(LEARNING_RATE),
                    "--weight-decay",
                    str(WEIGHT_DECAY),
                    "--warmup-ratio",
                    str(WARMUP_RATIO),
                    "--early-stopping-patience",
                    str(EARLY_STOPPING_PATIENCE),
                    "--seed",
                    str(SEED),
                    "--epochs",
                    str(NUM_EPOCHS),
                    "--max-length",
                    str(MAX_LENGTH),
                    "--per-device-batch-size",
                    str(PER_DEVICE_BATCH_SIZE),
                    "--gradient-accumulation-steps",
                    str(GRADIENT_ACCUMULATION_STEPS),
                    "--resume" if RESUME else "--no-resume",
                ]
                print("Commande complète:", full_training_command)
                subprocess.run(
                    full_training_command,
                    cwd=REPO_DIR,
                    check=True,
                    shell=False,
                )
            """
        ),
        _markdown(
            SECTION_TITLES[17],
            """
            Avec `RESUME=True`, le CLI cherche le dernier checkpoint complet compatible, vérifie
            checksums et fingerprints, puis restaure modèle, optimiseur, scheduler, scaler, RNG,
            époque et step. Un checkpoint incomplet/incompatible est refusé; aucun redémarrage
            silencieux à l’époque zéro n’est permis.
            """,
        ),
        _code(
            """
            import shlex
            import sys

            exact_resume_command = [
                sys.executable,
                "scripts/train_xlmr_multitask.py",
                "--config",
                "configs/xlmr_multitask_colab_v0.1.yaml",
                "--dataset",
                DATASET_DIR,
                "--output",
                CHECKPOINT_ROOT,
                "--reports",
                REPORT_ROOT,
                "--training-mode",
                TRAINING_MODE,
                "--model-name",
                MODEL_NAME,
                "--learning-rate",
                str(LEARNING_RATE),
                "--weight-decay",
                str(WEIGHT_DECAY),
                "--warmup-ratio",
                str(WARMUP_RATIO),
                "--early-stopping-patience",
                str(EARLY_STOPPING_PATIENCE),
                "--seed",
                str(SEED),
                "--epochs",
                str(NUM_EPOCHS),
                "--max-length",
                str(MAX_LENGTH),
                "--per-device-batch-size",
                str(PER_DEVICE_BATCH_SIZE),
                "--gradient-accumulation-steps",
                str(GRADIENT_ACCUMULATION_STEPS),
                "--resume",
            ]
            print("Commande exacte de reprise:")
            print(shlex.join(exact_resume_command))
            """
        ),
        _markdown(
            SECTION_TITLES[18],
            """
            Le CLI évalue automatiquement le meilleur checkpoint sélectionné par validation. Les
            résultats officiels restent séparés de la validation. Quand l’entraînement complet est
            désactivé, les cellules d’affichage utilisent les rapports du smoke test s’ils existent.
            """,
        ),
        _code(
            """
            import json
            from pathlib import Path

            try:
                import pandas as pd
            except ImportError:
                pd = None

            print("Chargement des métriques du meilleur checkpoint")
            full_manifest = Path(REPORT_ROOT) / "run_manifest.json"
            smoke_manifest = Path(REPORT_ROOT) / "smoke-test" / "run_manifest.json"
            if full_manifest.is_file() and json.loads(
                full_manifest.read_text(encoding="utf-8")
            ).get("status") == "COMPLETE":
                ACTIVE_REPORT_ROOT = Path(REPORT_ROOT)
                ACTIVE_CHECKPOINT_ROOT = Path(CHECKPOINT_ROOT)
            elif smoke_manifest.is_file():
                ACTIVE_REPORT_ROOT = Path(REPORT_ROOT) / "smoke-test"
                ACTIVE_CHECKPOINT_ROOT = Path(CHECKPOINT_ROOT) / "smoke-test"
            else:
                raise RuntimeError("Aucun rapport complet disponible; exécuter le smoke test")

            active_run_manifest = json.loads(
                (ACTIVE_REPORT_ROOT / "run_manifest.json").read_text(encoding="utf-8")
            )
            validation_metrics = json.loads(
                (ACTIVE_REPORT_ROOT / "validation_metrics.json").read_text(encoding="utf-8")
            )
            test_metrics = json.loads(
                (ACTIVE_REPORT_ROOT / "test_metrics.json").read_text(encoding="utf-8")
            )
            validation_rows = [
                {
                    "head": head,
                    "macro_f1": values.get("macro_f1"),
                    "accuracy": values.get("accuracy"),
                    "weighted_f1": values.get("weighted_f1"),
                }
                for head, values in validation_metrics.items()
                if isinstance(values, dict) and "macro_f1" in values
            ]
            print("Meilleur checkpoint:", active_run_manifest.get("best_checkpoint"))
            print("Core macro F1 validation:", validation_metrics.get("core_macro_f1"))
            if pd is not None:
                display(pd.DataFrame(validation_rows))
            else:
                print(validation_rows)
            ood_rows = []
            for split, split_values in test_metrics.items():
                ood_rows.append(
                    {
                        "split": split,
                        "head": "__core_macro_f1__",
                        "macro_f1": split_values.get("core_macro_f1"),
                    }
                )
                ood_rows.extend(
                    {
                        "split": split,
                        "head": head,
                        "macro_f1": values.get("macro_f1"),
                    }
                    for head, values in split_values.items()
                    if isinstance(values, dict) and "macro_f1" in values
                )
            if ood_rows:
                display(pd.DataFrame(ood_rows)) if pd is not None else print(ood_rows)
            else:
                print("Le smoke test ne produit pas d’évaluation OOD complète.")
            """
        ),
        _markdown(
            SECTION_TITLES[19],
            """
            Affichage textuel uniquement des sensibilités au changement, cohérences invariantes et
            exactitudes par tête; aucun texte candidat n’est chargé dans le rapport.
            """,
        ),
        _code(
            """
            import json

            counterfactual_path = ACTIVE_REPORT_ROOT / "counterfactual_results.json"
            counterfactual_results = json.loads(counterfactual_path.read_text(encoding="utf-8"))
            counterfactual_rows = [
                {
                    "head": head,
                    "pairwise_accuracy": values.get("pairwise_accuracy"),
                    "expected_change_sensitivity": values.get(
                        "expected_change_sensitivity"
                    ),
                    "invariant_prediction_consistency": values.get(
                        "invariant_prediction_consistency"
                    ),
                    "exact_group_accuracy": values.get("exact_group_accuracy"),
                }
                for head, values in counterfactual_results.items()
                if isinstance(values, dict) and "pairwise_accuracy" in values
            ]
            print("Analyse contrefactuelle:")
            if counterfactual_rows and pd is not None:
                display(pd.DataFrame(counterfactual_rows))
            else:
                print(counterfactual_rows or "Non disponible dans le smoke test")
            """
        ),
        _markdown(
            SECTION_TITLES[20],
            """
            Les faux positifs sont agrégés par split, langue et catégorie, notamment pour les cas
            cités, hypothétiques et français. Aucun payload complet n’est affiché.
            """,
        ),
        _code(
            """
            import json

            hard_negative_results = json.loads(
                (ACTIVE_REPORT_ROOT / "hard_negative_results.json").read_text(encoding="utf-8")
            )
            hard_negative_rows = []
            for split, split_values in hard_negative_results.items():
                for language, categories in split_values.get(
                    "by_language_and_category", {}
                ).items():
                    for category, values in categories.items():
                        hard_negative_rows.append(
                            {
                                "split": split,
                                "language": language,
                                "category": category,
                                "records": values.get("records"),
                                "false_positive_rate": values.get("false_positive_rate"),
                            }
                        )
            print("Négatifs difficiles:")
            if hard_negative_rows and pd is not None:
                display(pd.DataFrame(hard_negative_rows))
            else:
                print(hard_negative_rows or "Non disponible")
            """
        ),
        _markdown(
            SECTION_TITLES[21],
            """
            Les écarts anglais/français sont descriptifs: le split language-OOD peut différer par
            d’autres facteurs que la langue seule.
            """,
        ),
        _code(
            """
            import json

            language_results = json.loads(
                (ACTIVE_REPORT_ROOT / "language_results.json").read_text(encoding="utf-8")
            )
            language_rows = [
                {
                    "head": head,
                    "english_macro_f1": values.get("english_macro_f1"),
                    "french_macro_f1": values.get("french_macro_f1"),
                    "difference_en_minus_fr": values.get("difference_en_minus_fr"),
                }
                for head, values in language_results.get("heads", {}).items()
            ]
            print("Analyse anglais/français:")
            if language_rows and pd is not None:
                display(pd.DataFrame(language_rows))
            else:
                print(language_rows or "Non disponible dans le smoke test")
            print(language_results.get("caveat", ""))
            """
        ),
        _markdown(
            SECTION_TITLES[22],
            """
            Vérification de l’export `best_model/`, de ses métadonnées et de ses checksums. Le
            modèle reste sur Drive. L’exemple d’inférence est inoffensif et désactivé par défaut
            pour éviter un rechargement mémoire involontaire.
            """,
        ),
        _code(
            """
            import hashlib
            import json
            from pathlib import Path

            from promptsec.training.checkpoints import checkpoint_inventory

            print("Inventaire final des checkpoints et du modèle exporté")
            final_inventory = checkpoint_inventory(ACTIVE_CHECKPOINT_ROOT)
            print(
                "Checkpoints:",
                [(item["path"], item["status"]) for item in final_inventory["checkpoints"]],
            )
            best_model_root = ACTIVE_CHECKPOINT_ROOT / "best_model"
            if not best_model_root.is_dir():
                raise RuntimeError(f"Export best_model absent: {best_model_root}")
            expected_model_files = {
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "label_mappings.json",
                "classification_thresholds.json",
                "preprocessing_configuration.json",
                "dataset_fingerprint.json",
                "training_configuration.json",
                "validation_summary.json",
                "README.md",
                "checksums.json",
            }
            missing_model_files = sorted(
                name for name in expected_model_files if not (best_model_root / name).is_file()
            )
            if missing_model_files:
                raise RuntimeError(f"Fichiers du modèle exporté absents: {missing_model_files}")
            dataset_fingerprint = json.loads(
                (best_model_root / "dataset_fingerprint.json").read_text(encoding="utf-8")
            )
            training_configuration = json.loads(
                (best_model_root / "training_configuration.json").read_text(encoding="utf-8")
            )
            preprocessing_configuration = json.loads(
                (best_model_root / "preprocessing_configuration.json").read_text(
                    encoding="utf-8"
                )
            )
            if dataset_fingerprint.get("manifest_sha256") != bundle.manifest_sha256:
                raise RuntimeError("Le modèle exporté ne correspond pas au manifeste dataset")
            if not training_configuration.get("training_config_hash"):
                raise RuntimeError("Le hash de configuration d’entraînement est absent")
            if not preprocessing_configuration.get("special_tokens"):
                raise RuntimeError("Les tokens spéciaux du tokenizer sont absents")
            declared_checksums = json.loads(
                (best_model_root / "checksums.json").read_text(encoding="utf-8")
            )
            def model_file_sha256(path, chunk_size=8 * 1024 * 1024):
                digest = hashlib.sha256()
                with Path(path).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            for relative_name, expected_digest in declared_checksums.items():
                candidate = best_model_root / relative_name
                digest = model_file_sha256(candidate)
                if digest != expected_digest:
                    raise RuntimeError(f"Checksum modèle invalide: {relative_name}")
            drive_bytes = sum(
                path.stat().st_size
                for path in Path(ACTIVE_CHECKPOINT_ROOT).rglob("*")
                if path.is_file()
            )
            print("best_model vérifié:", best_model_root)
            print("Taille totale checkpoints/modèle sur Drive (GiB):", drive_bytes / 2**30)
            """
        ),
        _code(
            """
            if RUN_INFERENCE_EXAMPLE:
                from promptsec.training.inference import PromptSecPredictor

                print("Inférence synthétique inoffensive")
                predictor = PromptSecPredictor(best_model_root)
                harmless_context = {
                    "context": {
                        "protected_policy": "Agir uniquement avec l’autorisation de l’utilisateur.",
                        "user_goal": "Résumer un document public.",
                        "available_capabilities": ["READ_DOCUMENT"],
                    },
                    "content": {
                        "text": "Ceci est une phrase descriptive sans instruction opérationnelle.",
                        "source_role": "EXTERNAL_CONTENT",
                        "content_origin": "DOCUMENT",
                        "delivery_mode": "INDIRECT",
                        "ingestion_path": "RETRIEVAL",
                        "modality": "TEXT",
                        "source_integrity": "UNVERIFIED",
                    },
                }
                inference_result = predictor.predict(harmless_context)
                print(inference_result["derived_verdict"])
            else:
                print("Exemple d’inférence prêt; définir RUN_INFERENCE_EXAMPLE=True pour l’exécuter.")
            """
        ),
        _markdown(
            SECTION_TITLES[23],
            """
            Aperçu concis du runtime, de la mémoire GPU, des événements de reprise et des éventuels
            OOM récupérés. Les rapports JSON/CSV/Markdown complets restent sous `REPORT_ROOT`.
            """,
        ),
        _code(
            """
            import json

            resource_usage = json.loads(
                (ACTIVE_REPORT_ROOT / "resource_usage.json").read_text(encoding="utf-8")
            )
            active_manifest = json.loads(
                (ACTIVE_REPORT_ROOT / "run_manifest.json").read_text(encoding="utf-8")
            )
            report_preview = {
                "report_root": str(ACTIVE_REPORT_ROOT),
                "run_kind": active_manifest.get("run_kind"),
                "best_checkpoint": active_manifest.get("best_checkpoint"),
                "parameter_count": active_manifest.get("parameter_count"),
                "duration_seconds": resource_usage.get("duration_seconds"),
                "peak_gpu_memory_bytes": resource_usage.get("peak_gpu_memory_bytes"),
                "resolved_batch_strategy": resource_usage.get("resolved_batch_strategy"),
                "resume_events": active_manifest.get("resume_events", []),
                "oom_events": resource_usage.get("oom_events", []),
            }
            print("Aperçu des rapports:", report_preview)
            print("Rapport final:", ACTIVE_REPORT_ROOT / "final_report.md")
            print("Model card:", ACTIVE_REPORT_ROOT / "model_card.md")
            """
        ),
        _markdown(
            SECTION_TITLES[24],
            """
            Ce résumé capture GitHub, dataset, modèle, configuration résolue, meilleur checkpoint
            et commande de reprise exacte. Le SHA d’archive provient toujours du sidecar vérifié.
            """,
        ),
        _code(
            """
            import json
            import shlex
            import subprocess

            exact_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_DIR,
                check=True,
                shell=False,
                text=True,
                capture_output=True,
            ).stdout.strip()
            final_manifest = json.loads(
                (ACTIVE_REPORT_ROOT / "run_manifest.json").read_text(encoding="utf-8")
            )
            if final_manifest.get("source_commit_hash") != exact_commit:
                raise RuntimeError("Le run manifest ne correspond pas au commit Git exact")
            reproducibility_summary = {
                "repository_url": REPO_URL,
                "repository_commit": exact_commit,
                "dataset_archive_sha256": ARCHIVE_SHA256,
                "dataset_release_manifest_sha256": bundle.manifest_sha256,
                "model_name": MODEL_NAME,
                "training_mode": TRAINING_MODE,
                "seed": SEED,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "precision": PRECISION_MODE,
                "max_length": MAX_LENGTH,
                "epochs_requested": NUM_EPOCHS,
                "best_checkpoint": final_manifest.get("best_checkpoint"),
                "checkpoint_root": str(ACTIVE_CHECKPOINT_ROOT),
                "report_root": str(ACTIVE_REPORT_ROOT),
                "exact_resume_command": shlex.join(exact_resume_command),
            }
            print(json.dumps(reproducibility_summary, indent=2, ensure_ascii=False))
            print("The model was trained on synthetic SILVER labels.")
            print("This is not human-Gold evaluation.")
            print("Strong PolicyBench performance does not prove real-world robustness.")
            """
        ),
        _markdown(
            SECTION_TITLES[25],
            """
            `/content/PromptSec-FM` et `/content/promptsec_data` sont éphémères. Ils peuvent être
            supprimés après vérification des sorties Drive. Ne jamais supprimer automatiquement le
            ZIP, les checkpoints, les rapports ou le modèle dans Drive.
            """,
        ),
        _code(
            """
            import shutil
            from pathlib import Path

            if CLEAN_LOCAL_RUNTIME:
                safe_targets = [Path(REPO_DIR).resolve(), Path(LOCAL_DATA_ROOT).resolve()]
                content_root = Path("/content").resolve()
                drive_root = Path("/content/drive").resolve()
                for target in safe_targets:
                    if content_root not in target.parents or drive_root in target.parents:
                        raise RuntimeError(f"Cible de nettoyage refusée: {target}")
                print("Nettoyage des seules copies locales éphémères:", safe_targets)
                for target in safe_targets:
                    shutil.rmtree(target, ignore_errors=True)
            else:
                print(
                    "Nettoyage désactivé. Les données persistantes restent uniquement dans Drive."
                )
            """
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"promptsec-{index:03d}"
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
            "colab": {
                "name": "PromptSec_FM_XLMR_Multitask_Colab.ipynb",
                "provenance": [],
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    loaded = nbformat.read(args.output, as_version=4)
    nbformat.validate(loaded)
    print(f"Validated notebook: {args.output} ({len(loaded.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
