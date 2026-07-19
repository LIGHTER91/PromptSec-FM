# PromptSec-FM XLM-R multi-task Colab workflow v0.1

This workflow trains `FacebookAI/xlm-roberta-base` on the immutable 6,000-record
PromptSec-PolicyBench release. The annotations are synthetic `SILVER_VALIDATED`
labels, not human Gold truth. Good PolicyBench performance does not prove
real-world prompt-injection robustness.

The default `SCIENTIFIC_EVALUATION` mode trains only on the official `train`
split. Validation controls early stopping, thresholds, checkpoint selection,
and model selection. Policy-family, domain, language, and counterfactual tests
are evaluated once after selection. `human_review_candidates` is never trained
on in the scientific run, and no random record-level split is created.

## 1. Create the Colab archive locally

From PowerShell in the repository root:

```powershell
& .\.venv\Scripts\python.exe scripts\package_policybench_for_colab.py `
  --dataset data\generated\policybench-codex-v0.1 `
  --output artifacts\colab-input\policybench-codex-v0.1.zip
```

The command validates the complete source release, packages only the seven
official split views, release metadata, checksum index, and required schemas,
then checks that the source release remained byte-identical. Existing compatible
output is reused. Use `--overwrite` only when intentionally replacing it.

## 2. Locate and upload the files

The local outputs are:

- `artifacts\colab-input\policybench-codex-v0.1.zip`
- `artifacts\colab-input\policybench-codex-v0.1.zip.sha256`
- `artifacts\colab-input\colab_input_manifest.json`

Upload the ZIP and `.sha256` file to:

```text
/content/drive/MyDrive/PromptSec-FM/data/
```

The manifest is useful for local audit but is also embedded in the ZIP. Do not
upload raw attempts, caches, quarantine data, review annotations, credentials,
model artifacts, or operational logs.

## 3. Open the notebook and select a GPU

Open `notebooks/PromptSec_FM_XLMR_Multitask_Colab.ipynb` in Colab. Select
**Runtime → Change runtime type → GPU**. The notebook reports Python, PyTorch,
Transformers, CUDA, GPU, VRAM, bf16 support, system RAM, and measurable Drive
space. Full training aborts with an actionable error if CUDA is unavailable; it
never silently trains XLM-R on CPU.

## 4. Configure Drive and repository paths

Edit the single configuration cell near the top. `DRIVE_ROOT` defaults to:

```python
/content/drive/MyDrive/PromptSec-FM
```

Set `REPO_URL` only when the repository is not already available at
`/content/PromptSec-FM`. Do not put credentials or private Drive identifiers in
the notebook. No API key, OpenAI service, Codex process, Ollama server, W&B, or
MLflow service is required.

## 5. Verify and extract data

The notebook compares the archive SHA-256 before extraction and rejects path
traversal entries. It extracts to `/content/promptsec_data`, so training reads
the fast local Colab disk rather than a compressed Drive file. The loader then
validates package checksums, exact split counts, canonical IDs, SILVER state,
label vocabularies, and leakage constraints before downloading the model.

## 6. Execute the smoke test

Leave `RUN_SMOKE_TEST_FIRST = True`. The smoke command uses approximately 32
training records, 16 validation records, one epoch, and `max_length=128`. It
checks all nine heads, the backward pass, metrics, atomic checkpoint creation,
checkpoint reload, and a one-step resume probe. Smoke outputs are isolated under
`CHECKPOINT_ROOT/smoke-test` and `REPORT_ROOT/smoke-test`; they cannot resume into
the full run.

## 7. Start full scientific training

Leave `TRAINING_MODE = "SCIENTIFIC_EVALUATION"` and set
`RUN_FULL_TRAINING = True`. The notebook invokes:

```bash
python scripts/train_xlmr_multitask.py \
  --config configs/xlmr_multitask_colab_v0.1.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1 \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1 \
  --training-mode SCIENTIFIC_EVALUATION \
  --model-name FacebookAI/xlm-roberta-base \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.10 \
  --early-stopping-patience 2 \
  --epochs 4 \
  --max-length 512 \
  --seed 20260718 \
  --resume
```

At least 15 GiB VRAM resolves to batch 8 with accumulation 2; 10–15 GiB resolves
to batch 4 with accumulation 4; lower VRAM resolves to batch 2 with accumulation
8. The target effective batch size is 16. bf16 is selected only when supported,
otherwise fp16 is used. Gradient checkpointing, dynamic padding, gradient norm
1.0, and zero unnecessary dataloader workers are enabled.

## 8. Safely stop and resume

Wait for a checkpoint cell or epoch checkpoint to finish before deliberately
stopping a session. Checkpoints are written to a temporary sibling directory,
validated, checksummed, and renamed only after completion. On reconnect, remount
Drive, rerun setup and verification cells, and execute the same command with
`--resume`. Complete checkpoints are listed before training. Dataset,
configuration, label, special-token, and run-kind fingerprints must match.
Incomplete or incompatible checkpoints are rejected; resume never silently
restarts at epoch zero.

If CUDA runs out of memory, the failed attempt is reported, the cache is
cleared, batch size is halved, accumulation is doubled, and the run retries once.
There is no infinite retry loop.

## 9. Locate checkpoints and reports

Checkpoints:

```text
/content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1/
```

Reports:

```text
/content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1/
```

The selected standalone export is under `best_model/` and includes safetensors,
tokenizer, section tokens, frozen mappings, thresholds, preprocessing, dataset
and training fingerprints, validation summary, model card, and checksums. It
does not include training records.

## 10. Final SILVER model mode

`FINAL_SILVER_MODEL` is disabled by default and requires an explicit
`final_silver_splits` pool in a separate configuration. Any reused evaluation
records cease to provide independent test performance and are never reported as
unbiased. Complete scientific evaluation first and use a separate output path.

## 11. Optional Hugging Face workflow

The notebook exposes an optional interactive login switch but performs no Hub
upload. It never stores a token. If a user later implements an upload, only the
selected `best_model` may be uploaded—not the dataset, credentials, raw
generations, hidden review metadata, or unselected checkpoints.

## 12. Inference

`promptsec.training.inference.PromptSecPredictor` loads the exported directory
without the original training process. Its `predict()` method accepts one
canonical context record and returns decoded single-label predictions,
probabilities, multi-label predictions and probabilities, derived verdict
information, and model/preprocessing versions.

```python
from promptsec.training.inference import PromptSecPredictor

predictor = PromptSecPredictor(
    "/content/drive/MyDrive/PromptSec-FM/checkpoints/"
    "xlmr-base-multitask-v0.1/best_model"
)
result = predictor.predict(
    {
        "context": {
            "protected_policy": "Only act with current-user authorization.",
            "user_goal": "Summarize the retrieved document.",
            "available_capabilities": ["READ_DOCUMENT"],
        },
        "content": {
            "text": "The candidate content to classify.",
            "source_role": "EXTERNAL_CONTENT",
            "content_origin": "DOCUMENT",
            "delivery_mode": "INDIRECT",
            "ingestion_path": "RETRIEVAL",
            "modality": "TEXT",
            "source_integrity": "UNVERIFIED",
        },
    }
)
print(result["derived_verdict"])
```

## 13. Spans and interpretation

Existing spans remain untouched in the release. Span extraction is a separate
future model phase and is not approximated by this classifier. Evaluate all
metrics as synthetic SILVER performance. In particular, English/French gaps are
descriptive because the official language-OOD split may differ in more than
language alone.
