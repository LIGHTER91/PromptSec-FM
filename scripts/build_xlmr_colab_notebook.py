#!/usr/bin/env python3
# ruff: noqa: E501
"""Build and validate the executable PromptSec-FM XLM-R Colab notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "notebooks" / "PromptSec_FM_XLMR_Multitask_Colab.ipynb"


def _markdown(title: str, content: str) -> Any:
    return new_markdown_cell(f"## {title}\n\n{content}")


def build_notebook() -> Any:
    cells = [
        new_markdown_cell(
            "# PromptSec-FM XLM-R Multitask Training\n\n"
            "This workflow trains on synthetic, automatically validated **SILVER** labels. "
            "They are not human Gold truth, and good PolicyBench scores do not prove "
            "real-world prompt-injection robustness. Full training requires a CUDA GPU."
        ),
        _markdown(
            "1. Project overview and SILVER warning",
            "Nine frozen taxonomy heads share one multilingual XLM-R encoder. Span prediction is intentionally deferred to a future phase.",
        ),
        _markdown(
            "2. Google Drive mounting",
            "Authorize only the Drive account that owns the configured PromptSec-FM directory.",
        ),
        new_code_cell("from google.colab import drive\n\ndrive.mount('/content/drive')"),
        _markdown(
            "3. User configuration",
            "Edit this one cell before running the notebook. No private Drive identifier or credential is hardcoded.",
        ),
        new_code_cell(
            "PROJECT_NAME = 'PromptSec-FM'\n"
            "RUN_NAME = 'xlmr-base-multitask-v0.1'\n"
            "MODEL_NAME = 'FacebookAI/xlm-roberta-base'\n"
            "REPO_URL = ''  # Optional public/fork Git URL; leave empty if REPO_DIR already exists.\n"
            "REPO_DIR = '/content/PromptSec-FM'\n"
            "DRIVE_ROOT = '/content/drive/MyDrive/PromptSec-FM'\n"
            "DATA_ARCHIVE = f'{DRIVE_ROOT}/data/policybench-codex-v0.1.zip'\n"
            "DATA_SHA256_FILE = f'{DATA_ARCHIVE}.sha256'\n"
            "DATA_ROOT = '/content/promptsec_data/policybench-codex-v0.1'\n"
            "CHECKPOINT_ROOT = f'{DRIVE_ROOT}/checkpoints/{RUN_NAME}'\n"
            "REPORT_ROOT = f'{DRIVE_ROOT}/reports/{RUN_NAME}'\n"
            "TRAINING_MODE = 'SCIENTIFIC_EVALUATION'\n"
            "MAX_LENGTH = 512\nNUM_EPOCHS = 4\nLEARNING_RATE = 2e-5\n"
            "WEIGHT_DECAY = 0.01\nWARMUP_RATIO = 0.10\nEARLY_STOPPING_PATIENCE = 2\n"
            "SEED = 20260718\nRESUME = True\nRUN_SMOKE_TEST_FIRST = True\n"
            "RUN_FULL_TRAINING = True\nENABLE_HF_LOGIN = False"
        ),
        _markdown(
            "4. GPU and runtime preflight",
            "The full run aborts rather than silently using CPU. A CPU smoke test is permitted only through the explicit smoke flag.",
        ),
        new_code_cell(
            "import importlib.metadata\nimport os\nimport platform\nimport shutil\n"
            "from pathlib import Path\n\nimport torch\n\n"
            "print('Python:', platform.python_version())\nprint('PyTorch:', torch.__version__)\n"
            "try:\n    print('Transformers:', importlib.metadata.version('transformers'))\n"
            "except importlib.metadata.PackageNotFoundError:\n    print('Transformers: not installed yet')\n"
            "print('CUDA available:', torch.cuda.is_available())\nprint('CUDA version:', torch.version.cuda)\n"
            "if torch.cuda.is_available():\n"
            "    free, total = torch.cuda.mem_get_info()\n"
            "    print('GPU:', torch.cuda.get_device_name(0))\n"
            "    print('VRAM total/free GiB:', total/2**30, free/2**30)\n"
            "    print('bf16 supported:', torch.cuda.is_bf16_supported())\n"
            "    print('Selected precision:', 'bf16' if torch.cuda.is_bf16_supported() else 'fp16')\n"
            "else:\n    print('GPU: none; full training will abort, explicit smoke test may use CPU')\n"
            "page_size = os.sysconf('SC_PAGE_SIZE')\npages = os.sysconf('SC_PHYS_PAGES')\n"
            "available_pages = os.sysconf('SC_AVPHYS_PAGES')\n"
            "print(\n    'System RAM total/available GiB:',\n    page_size * pages / 2**30,\n    page_size * available_pages / 2**30,\n)\n"
            "if Path(DRIVE_ROOT).exists():\n"
            "    print('Drive free GiB:', shutil.disk_usage(DRIVE_ROOT).free / 2**30)"
        ),
        _markdown(
            "5. Repository setup",
            "Clone only when the repository is not already present in the Colab runtime.",
        ),
        new_code_cell(
            "import os\nimport subprocess\nfrom pathlib import Path\n\n"
            "if not Path(REPO_DIR).is_dir():\n"
            "    if not REPO_URL:\n        raise RuntimeError('Set REPO_URL or upload/mount the repository at REPO_DIR.')\n"
            "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)\n"
            "os.chdir(REPO_DIR)\nprint('Repository:', Path.cwd())"
        ),
        _markdown(
            "6. Dependency installation",
            "Install the repository's bounded training extra; no experiment-tracking service or API key is used.",
        ),
        new_code_cell("%pip install -q -e '.[training]'"),
        _markdown(
            "7. Dataset archive verification",
            "The uploaded ZIP is rejected before extraction when its SHA-256 differs.",
        ),
        new_code_cell(
            "import hashlib\nfrom pathlib import Path\n\n"
            "archive = Path(DATA_ARCHIVE)\nsha_file = Path(DATA_SHA256_FILE)\n"
            "if not archive.is_file() or not sha_file.is_file():\n    raise FileNotFoundError('Upload both ZIP and .sha256 files to DRIVE_ROOT/data.')\n"
            "expected = sha_file.read_text(encoding='utf-8').split()[0].lower()\n"
            "digest = hashlib.sha256(archive.read_bytes()).hexdigest()\n"
            "if digest != expected:\n    raise RuntimeError(f'Archive SHA-256 mismatch: {digest} != {expected}')\n"
            "print('Verified archive SHA-256:', digest)"
        ),
        _markdown(
            "8. Dataset extraction",
            "Training uses the local Colab disk, not the compressed Drive file.",
        ),
        new_code_cell(
            "import shutil\nimport zipfile\nfrom pathlib import Path\n\n"
            "extract_root = Path('/content/promptsec_data')\n"
            "if extract_root.exists():\n    shutil.rmtree(extract_root)\n"
            "extract_root.mkdir(parents=True)\n"
            "with zipfile.ZipFile(DATA_ARCHIVE) as zf:\n"
            "    root = extract_root.resolve()\n"
            "    for member in zf.infolist():\n"
            "        target = (root / member.filename).resolve()\n"
            "        if root not in target.parents and target != root:\n"
            "            raise RuntimeError('Unsafe ZIP path')\n"
            "    zf.extractall(root)\nprint('Extracted:', DATA_ROOT)"
        ),
        _markdown(
            "9. Release and split integrity checks",
            "This performs packaged checksums, canonical state, exact split counts, ID uniqueness, and leakage gates before model loading.",
        ),
        new_code_cell(
            "from promptsec.training.dataset import load_training_dataset\n\n"
            "bundle = load_training_dataset(DATA_ROOT)\n"
            "print(bundle.integrity_report)\n"
            "assert bundle.integrity_report['automatic_gold_records'] == 0"
        ),
        _markdown(
            "10. Label-vocabulary inspection",
            "Mappings come from the packaged frozen schema and are stored with hashes.",
        ),
        new_code_cell(
            "for head, mapping in bundle.mappings.items():\n    print(head, mapping.labels, mapping.mapping_hash)"
        ),
        _markdown(
            "11. Tokenization and serialization preview using redacted examples",
            "Only section token lengths and a text hash are displayed; attack payloads are not printed.",
        ),
        new_code_cell(
            "from transformers import AutoTokenizer\n\n"
            "from promptsec.training.serialization import (\n"
            "    SPECIAL_TOKENS,\n    record_sections,\n    serialize_full_context,\n)\n\n"
            "sample = bundle.records_by_split['train'][0]\nserialized = serialize_full_context(sample)\n"
            "preview_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)\n"
            "preview_tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS)})\n"
            "print('record_id:', sample['id'])\nprint('serialized_sha256:', hashlib.sha256(serialized.encode()).hexdigest())\n"
            "section_token_lengths = {\n"
            "    name: len(preview_tokenizer.encode(value, add_special_tokens=False))\n"
            "    for name, value in record_sections(sample).as_ordered_items()\n}\n"
            "print('section_token_lengths:', section_token_lengths)"
        ),
        _markdown(
            "12. Optional small smoke test",
            "Uses 32 training and 16 validation records, one epoch, length 128, separate checkpoints, backward pass, checkpoint reload, and one-step resume probe.",
        ),
        new_code_cell(
            "import subprocess\n\n"
            "base_cmd = [\n"
            "    'python',\n    'scripts/train_xlmr_multitask.py',\n    '--config',\n"
            "    'configs/xlmr_multitask_colab_v0.1.yaml',\n    '--dataset',\n    DATA_ROOT,\n"
            "    '--output',\n    CHECKPOINT_ROOT,\n    '--reports',\n    REPORT_ROOT,\n"
            "    '--training-mode',\n    TRAINING_MODE,\n    '--model-name',\n    MODEL_NAME,\n"
            "    '--learning-rate',\n    str(LEARNING_RATE),\n    '--weight-decay',\n    str(WEIGHT_DECAY),\n"
            "    '--warmup-ratio',\n    str(WARMUP_RATIO),\n    '--early-stopping-patience',\n"
            "    str(EARLY_STOPPING_PATIENCE),\n    '--seed',\n    str(SEED),\n]\n"
            "if RUN_SMOKE_TEST_FIRST:\n"
            "    smoke_cmd = base_cmd + [\n"
            "        '--smoke-test',\n        '--max-train-records',\n        '32',\n"
            "        '--max-validation-records',\n        '16',\n        '--epochs',\n        '1',\n"
            "        '--max-length',\n        '128',\n        '--resume' if RESUME else '--no-resume',\n    ]\n"
            "    subprocess.run(smoke_cmd, check=True)"
        ),
        _markdown(
            "13. Full training",
            "Runs the tested CLI. SCIENTIFIC_EVALUATION optimizes only the official train split and selects using validation only.",
        ),
        new_code_cell(
            "from promptsec.training.checkpoints import checkpoint_inventory\n\n"
            "before_training = checkpoint_inventory(CHECKPOINT_ROOT)\n"
            "detected = [(item['path'], item['status']) for item in before_training['checkpoints']]\n"
            "print('Detected checkpoints:', detected)\n"
            "if RUN_FULL_TRAINING:\n"
            "    if not torch.cuda.is_available():\n"
            "        raise RuntimeError('Select a Colab GPU runtime before full training.')\n"
            "    full_cmd = base_cmd + [\n"
            "        '--epochs',\n        str(NUM_EPOCHS),\n        '--max-length',\n"
            "        str(MAX_LENGTH),\n        '--resume' if RESUME else '--no-resume',\n    ]\n"
            "    print('Training command:', ' '.join(full_cmd))\n    subprocess.run(full_cmd, check=True)"
        ),
        _markdown(
            "14. Automatic resume from Drive",
            "The command above uses `--resume`. It verifies complete checkpoint checksums and fingerprints; incompatible state is rejected instead of silently restarting.",
        ),
        new_code_cell(
            "from promptsec.training.checkpoints import checkpoint_inventory\n\n"
            "inventory = checkpoint_inventory(CHECKPOINT_ROOT)\n"
            "[(item['path'], item['status']) for item in inventory['checkpoints']]"
        ),
        _markdown(
            "15. Validation metrics",
            "Validation controls early stopping, checkpoint selection, and optional multi-label threshold selection.",
        ),
        new_code_cell(
            "import json\n\njson.loads(Path(REPORT_ROOT, 'validation_metrics.json').read_text())"
        ),
        _markdown(
            "16. Official test evaluation",
            "The selected checkpoint is evaluated once on policy, domain, language, and counterfactual tests.",
        ),
        new_code_cell("json.loads(Path(REPORT_ROOT, 'test_metrics.json').read_text())"),
        _markdown(
            "17. Counterfactual evaluation",
            "Includes pairwise, expected-change, invariant, exact-group, transition, and type-specific results.",
        ),
        new_code_cell("json.loads(Path(REPORT_ROOT, 'counterfactual_results.json').read_text())"),
        _markdown(
            "18. Language analysis",
            "EN/FR differences are descriptive because the language-OOD split may differ in other ways.",
        ),
        new_code_cell("json.loads(Path(REPORT_ROOT, 'language_results.json').read_text())"),
        _markdown(
            "19. Hard-negative analysis",
            "False positives are grouped by split, language, and category without printing payloads.",
        ),
        new_code_cell("json.loads(Path(REPORT_ROOT, 'hard_negative_results.json').read_text())"),
        _markdown(
            "20. Checkpoint export",
            "The selected standalone model is under `best_model`. Hugging Face login is optional and disabled; no automatic upload occurs.",
        ),
        new_code_cell(
            "best_model = Path(CHECKPOINT_ROOT, 'best_model')\nprint('Best model:', best_model)\n"
            "if ENABLE_HF_LOGIN:\n    from huggingface_hub import login\n    login()\n"
            "print('No Hub upload is performed by this notebook.')"
        ),
        _markdown(
            "21. Model card generation",
            "The CLI writes the model card locally on Drive with SILVER limitations.",
        ),
        new_code_cell("print(Path(REPORT_ROOT, 'model_card.md').read_text(encoding='utf-8'))"),
        _markdown(
            "22. Final run summary",
            "Review integrity, effective settings, resource usage, resume events, and OOM recovery.",
        ),
        new_code_cell("print(Path(REPORT_ROOT, 'final_report.md').read_text(encoding='utf-8'))"),
        _markdown(
            "23. Reproduction instructions",
            "Re-run the full command printed in section 13 with the same archive, configuration, and `--resume`. See `docs/xlmr_multitask_colab_v0.1.md` for PowerShell upload preparation and disconnect recovery.",
        ),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
            "colab": {"name": "PromptSec_FM_XLMR_Multitask_Colab.ipynb", "provenance": []},
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
