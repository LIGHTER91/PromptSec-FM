#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the isolated PromptSec-FM XLM-R v0.2 Colab notebook deterministically."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "PromptSec_FM_XLMR_Multitask_Colab_v0_2.ipynb"


def markdown(title: str, body: str) -> object:
    return nbformat.v4.new_markdown_cell(f"## {title}\n\n{body}")


def code(source: str) -> object:
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook() -> object:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# PromptSec-FM — XLM-R multitâche v0.2\n\n"
            "Workflow scientifique isolé. Il ne modifie ni v0.1, ni la taxonomie gelée, "
            "ni les 6 000 records SILVER. L’entraînement complet est désactivé par défaut."
        ),
        markdown(
            "1. Paramètres sûrs",
            "Les sorties v0.2 sont distinctes. Le français OOD ne sert jamais à la sélection.",
        ),
        code(
            """REPOSITORY_URL = "https://github.com/LIGHTER91/PromptSec-FM.git"
REPOSITORY_REF = "main"
REPO_DIR = "/content/PromptSec-FM"
DRIVE_ROOT = "/content/drive/MyDrive/PromptSec-FM"
DATASET_ARCHIVE = f"{DRIVE_ROOT}/data/policybench-codex-v0.1-colab.zip"
DATASET_DIR = "/content/promptsec_data/policybench-codex-v0.1"
V0_1_REPORT_ROOT = f"{DRIVE_ROOT}/reports/xlmr-base-multitask-v0.1"
START_V0_2_TRAINING = False
RUN_SMOKE_TEST_FIRST = True
RESET_SMOKE_TEST = True
RESUME_SMOKE_TEST = False
RESUME = True
RUN_BALANCED_EXPERIMENT = True
RUN_RELATIONAL_EXPERIMENT = True
RUN_FOCAL_EXPERIMENT = False
BILINGUAL_FINAL_SILVER_MODEL = False
PRUNE_CHECKPOINTS = False"""
        ),
        markdown("2. Montage de Drive", "Drive contient l’archive vérifiée et les artefacts."),
        code("from google.colab import drive\n\ndrive.mount('/content/drive')"),
        markdown("3. Clone complet", "Le dépôt complet est cloné; aucun sous-ensemble manuel."),
        code(
            """import subprocess
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(
        ["git", "clone", "--branch", REPOSITORY_REF, REPOSITORY_URL, REPO_DIR],
        check=True,
    )
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print("Commit utilisé:", commit)"""
        ),
        markdown("4. Installation", "Installation éditable avec le groupe `training`."),
        code(
            """import importlib
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[training]"],
    cwd=REPO_DIR,
    check=True,
)
load_training_dataset = importlib.import_module(
    "promptsec.training.dataset"
).load_training_dataset
print("Import PromptSec OK")"""
        ),
        markdown("5. Vérification de l’archive", "Le SHA-256 et le manifeste Colab sont vérifiés."),
        code(
            """import hashlib
import zipfile

archive = Path(DATASET_ARCHIVE)
if not archive.is_file():
    raise FileNotFoundError(archive)
print("SHA-256 archive:", hashlib.sha256(archive.read_bytes()).hexdigest())
extract_root = Path(DATASET_DIR).parent
if not Path(DATASET_DIR).exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(extract_root)"""
        ),
        markdown(
            "6. Validation des 6 000 records",
            "Le chargeur bloque tout écart SILVER, split ou checksum.",
        ),
        code(
            """bundle = load_training_dataset(DATASET_DIR)
print(bundle.integrity_report)
assert bundle.integrity_report["records"] == 6000
assert bundle.integrity_report["automatic_gold_records"] == 0"""
        ),
        markdown("7. Audit v0.1", "Les poids v0.1 ne sont pas nécessaires à la comparaison."),
        code(
            """from promptsec.training.diagnostics_v02 import v01_diagnostic_audit

v01_audit = v01_diagnostic_audit(V0_1_REPORT_ROOT)
print(v01_audit["status"], v01_audit.get("available_files", []))"""
        ),
        markdown(
            "8. Audit contrefactuel v0.2",
            "Seules les paires complètes du train officiel sont éligibles.",
        ),
        code(
            """from promptsec.training.pairing import audit_counterfactual_groups

pair_audit = audit_counterfactual_groups(bundle.records_by_split)
print(pair_audit["splits"].get("train"))
print(pair_audit["splits"].get("validation"))"""
        ),
        markdown(
            "9. Distributions",
            "La catégorie sert au sampling et au diagnostic, jamais à l’entrée modèle.",
        ),
        code(
            """from promptsec.training.diagnostics_v02 import (
    category_distribution,
    class_and_multilabel_audit,
)

print(category_distribution(bundle.records_by_split["train"]))
training_audit = class_and_multilabel_audit(
    bundle.records_by_split["train"], bundle.mappings
)
print(training_audit["heads"])"""
        ),
        markdown("10. Commandes communes", "Toutes les commandes appellent le CLI testé."),
        code(
            """CONFIG = "configs/xlmr_multitask_colab_v0.2.yaml"
def run_command(command):
    print("Commande:", command)
    subprocess.run(command, cwd=REPO_DIR, check=True, shell=False)

def experiment_command(name, *, resume=True):
    command = [sys.executable, "scripts/train_xlmr_multitask.py", "--config", CONFIG,
            "--dataset", DATASET_DIR,
            "--output", f"{DRIVE_ROOT}/checkpoints/xlmr-base-multitask-v0.2-{name}",
            "--reports", f"{DRIVE_ROOT}/reports/xlmr-base-multitask-v0.2-{name}",
            "--experiment", name, "--v0-1-report-root", V0_1_REPORT_ROOT,
            "--resume" if resume else "--no-resume"]
    if PRUNE_CHECKPOINTS:
        command.append("--prune-checkpoints")
    return command"""
        ),
        markdown("11. Smoke propre Experiment B", "Par défaut il démarre propre et sans reprise."),
        code(
            """if RUN_SMOKE_TEST_FIRST:
    smoke = experiment_command("relational", resume=RESUME_SMOKE_TEST)
    smoke += ["--smoke-test", "--max-train-records", "32", "--max-validation-records", "16",
              "--epochs", "1", "--max-length", "128"]
    if RESET_SMOKE_TEST:
        smoke.append("--reset-smoke-test")
    run_command(smoke)"""
        ),
        markdown(
            "12. Garde d’exécution",
            "Le smoke peut tourner; les runs complets exigent une activation explicite.",
        ),
        code('print("START_V0_2_TRAINING =", START_V0_2_TRAINING)'),
        markdown(
            "13. Experiment A — Balanced",
            "Aucune perte relationnelle; calibration multilabel validation-only.",
        ),
        code(
            """if START_V0_2_TRAINING and RUN_BALANCED_EXPERIMENT:
    run_command(experiment_command("balanced", resume=RESUME))"""
        ),
        markdown(
            "14. Experiment B — Relational",
            "Paires, JS tête par tête et cohérence structurée du verdict.",
        ),
        code(
            """if START_V0_2_TRAINING and RUN_RELATIONAL_EXPERIMENT:
    run_command(experiment_command("relational", resume=RESUME))"""
        ),
        markdown(
            "15. Experiment C — Focal optionnel",
            "L’asymmetric focal BCE est désactivée par défaut.",
        ),
        code(
            """if START_V0_2_TRAINING and RUN_FOCAL_EXPERIMENT:
    run_command(experiment_command("focal", resume=RESUME))"""
        ),
        markdown(
            "16. Évaluation sélectionnée",
            "Les tests officiels ne sont évalués qu’après le checkpoint validation.",
        ),
        code(
            """print("Les rapports test_metrics.json sont écrits uniquement après la sélection validation.")"""
        ),
        markdown(
            "17. Diagnostics relationnels",
            "Affichage sans payload : contrefactuels, sur-défense, logique et seuils.",
        ),
        code(
            """import json


def show_report(root, name):
    path = Path(root) / name
    print(name, json.loads(path.read_text()) if path.is_file() else "indisponible")
selected_reports = f"{DRIVE_ROOT}/reports/xlmr-base-multitask-v0.2-relational"
for name in ["counterfactual_results.json", "hard_negative_results.json",
             "multilabel_thresholds.json", "logical_consistency_results.json"]:
    show_report(selected_reports, name)"""
        ),
        markdown(
            "18. Diagnostics EN/FR",
            "Le français OOD est consulté seulement après sélection; les splits diffèrent au-delà de la langue.",
        ),
        code('show_report(selected_reports, "language_results.json")'),
        markdown(
            "19. Vérification des exports",
            "Les checksums de `best_model` doivent être valides avant tout pruning.",
        ),
        code(
            """from promptsec.training.retention import verify_export

best_export = f"{DRIVE_ROOT}/checkpoints/xlmr-base-multitask-v0.2-relational/best_model"
if Path(best_export).is_dir():
    print(verify_export(best_export))"""
        ),
        markdown(
            "20. Rétention optionnelle",
            "Le défaut reste un dry-run; v0.1 est hors périmètre par construction.",
        ),
        code(
            """print("PRUNE_CHECKPOINTS =", PRUNE_CHECKPOINTS)
print("Le CLI écrit checkpoint_pruning_manifest.json après vérification de l’export.")"""
        ),
        markdown(
            "21. Reproduction exacte",
            "Copiez les listes ci-dessous; aucune commande v0.1 n’est relancée.",
        ),
        code(
            """print(
    "Smoke:",
    experiment_command("relational", resume=False)
    + ["--smoke-test", "--reset-smoke-test"],
)
print("A:", experiment_command("balanced", resume=True))
print("B:", experiment_command("relational", resume=True))
print("C optionnel:", experiment_command("focal", resume=True))"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"promptsec-v02-{index:02d}"
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": OUTPUT.name, "provenance": []},
    }
    return notebook


def main() -> int:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(OUTPUT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    nbformat.read(OUTPUT, as_version=4)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
