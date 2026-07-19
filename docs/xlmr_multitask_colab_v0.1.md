# PromptSec-FM — entraînement XLM-R multi-tâche dans Google Colab

Ce workflow ouvre le notebook depuis GitHub, clone le dépôt complet dans le
runtime Colab, puis utilise Google Drive uniquement pour les données et sorties
persistantes. Il ne faut téléverser le dépôt ni dans Colab ni dans Drive.

Les 6 000 annotations PolicyBench sont synthétiques et `SILVER_VALIDATED`, pas
human-Gold. Les scores ne prouvent pas une robustesse en conditions réelles. Le
mode `SCIENTIFIC_EVALUATION` entraîne uniquement sur le split officiel `train`;
la validation sélectionne le checkpoint et les tests OOD ne servent jamais à
la sélection.

## A. Préparation sur la machine locale

### 1. Commit et push du code source vers GitHub

Vérifier les changements et ne jamais ajouter archive, dataset généré,
checkpoint ou poids de modèle:

```powershell
git status --short
git add .gitignore pyproject.toml configs/xlmr_multitask_colab_v0.1.yaml `
  docs/xlmr_multitask_colab_v0.1.md `
  notebooks/PromptSec_FM_XLMR_Multitask_Colab.ipynb `
  scripts/build_xlmr_colab_notebook.py `
  scripts/package_policybench_for_colab.py `
  scripts/train_xlmr_multitask.py scripts/evaluate_xlmr_multitask.py `
  src/promptsec/training tests/training
git diff --cached --stat
git commit -m "Add GitHub-first XLM-R Colab workflow"
git push origin main
```

Adapter la branche et la sélection des fichiers au worktree réel. Le workflow
ne commit et ne push rien automatiquement.

### 2. Construire l’archive PolicyBench

Depuis `C:/Users/lulup/Documents/PromptSec-FM`:

```powershell
& ./.venv/Scripts/python.exe scripts/package_policybench_for_colab.py `
  --dataset data/generated/policybench-codex-v0.1 `
  --output artifacts/colab-input/policybench-codex-v0.1.zip
```

La commande valide la release, les 6 000 records et les groupes
contrefactuels, construit un ZIP déterministe et vérifie que la release source
reste byte-identique. Une archive compatible est réutilisée; `--overwrite` est
requis pour la remplacer explicitement.

Fichiers locaux:

```text
artifacts/colab-input/policybench-codex-v0.1.zip
artifacts/colab-input/policybench-codex-v0.1.zip.sha256
artifacts/colab-input/colab_input_manifest.json
```

Hash actuel:

```text
0ccb70cb1db7c38ad52ba0c395a5b2a62a72d57870ca1af86711037c2e9a59e5
```

Le sidecar reste la source de vérité opérationnelle.

### 3. Téléverser uniquement le ZIP et son sidecar dans Drive

```text
MyDrive/
└── PromptSec-FM/
    ├── data/
    │   ├── policybench-codex-v0.1.zip
    │   └── policybench-codex-v0.1.zip.sha256
    ├── checkpoints/
    └── reports/
```

Ne pas téléverser le dépôt, les tentatives brutes, caches, credentials,
quarantaine ou annotations de revue.

## B. Exécution dans Google Colab

### 1. Ouvrir le notebook depuis GitHub

Dans l’onglet **GitHub** de Colab, chercher le dépôt public
`LIGHTER91/PromptSec-FM`, puis ouvrir:

```text
notebooks/PromptSec_FM_XLMR_Multitask_Colab.ipynb
```

Choisir **Runtime → Change runtime type → GPU** avant `Run all`.

### 2. Configurer GitHub et Drive

Valeurs par défaut:

```python
GITHUB_OWNER = "LIGHTER91"
GITHUB_REPOSITORY = "PromptSec-FM"
GITHUB_REF = "main"
REPO_DIR = "/content/PromptSec-FM"
DRIVE_ROOT = "/content/drive/MyDrive/PromptSec-FM"
START_FULL_TRAINING = False
```

Pour figer le code, remplacer `GITHUB_REF` par un tag ou le SHA exact d’un
commit poussé. Une branche est mise à jour uniquement par fast-forward; un tag
ou SHA est checkout en `DETACHED_HEAD`.

### 3. Préflight GPU

Le notebook affiche Python, système, PyTorch, Transformers, CUDA, runtime CUDA,
GPU, VRAM totale/libre, support bf16, RAM et espace Drive mesurable.

- VRAM ≥ 15 GiB: batch 8, accumulation 2;
- 10 ≤ VRAM < 15 GiB: batch 4, accumulation 4;
- VRAM < 10 GiB: batch 2, accumulation 8.

Le batch effectif vaut 16. bf16 est utilisé quand disponible, sinon fp16.
L’entraînement complet s’arrête sans CUDA. Un smoke CPU exige
`ALLOW_CPU_SMOKE_TEST=True` et sa vitesse n’est pas représentative.

### 4. Monter Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Le notebook crée uniquement `DRIVE_ROOT/data`, `CHECKPOINT_ROOT` et
`REPORT_ROOT`.

### 5. Cloner ou mettre à jour le dépôt

Premier passage, branche publique par défaut:

```bash
git clone --branch main --single-branch \
  https://github.com/LIGHTER91/PromptSec-FM.git \
  /content/PromptSec-FM
```

Si le dépôt existe, le notebook vérifie `.git`, l’URL `origin` et un worktree
propre, fetch `origin`, checkout la ref et applique `merge --ff-only` pour une
branche. Il refuse de détruire des changements locaux. `FORCE_RECLONE=True`
supprime uniquement `REPO_DIR` sous `/content`, jamais Drive.

Chemin, branche ou `DETACHED_HEAD`, SHA exact, remote, dernier commit et statut
du worktree sont affichés. Tous les imports/scripts proviennent de ce clone.

### 6. Installer le dépôt cloné

Après inspection de `pyproject.toml`, le notebook lance sans `--upgrade`:

```bash
python -m pip install -e ".[training]"
```

Avec `INSTALL_DEV_DEPENDENCIES=True`, la cible devient `.[training,dev]`.
L’absence de `--upgrade` conserve le PyTorch CUDA de Colab s’il satisfait les
bornes. Les versions runtime sont affichées et `promptsec.training` doit se
résoudre depuis `/content/PromptSec-FM/src`. Comme l’installation éditable est
lancée dans un sous-processus alors que le kernel Colab tourne déjà, la cellule
ajoute explicitement ce répertoire source au début de `sys.path` et invalide les
caches d’import ; aucun redémarrage du runtime n’est nécessaire.

### 7. Vérifier et extraire le dataset

Le SHA-256 est lu dans le sidecar puis calculé en streaming. Une divergence
bloque l’extraction. Le ZIP est extrait sous:

```text
/content/promptsec_data/policybench-codex-v0.1
```

Chemins absolus, traversées `..`, sorties de racine et symlinks sont refusés.
Une extraction compatible est réutilisée après vérification du manifeste et de
tous ses fichiers. Seule la copie locale éphémère est recréée; le ZIP Drive
n’est jamais supprimé.

Le loader vérifie 6 000 records, 3 000 EN/3 000 FR, 720 groupes
contrefactuels, 17 checksums, sept splits, identifiants uniques, isolation
anti-fuite, taxonomie gelée et état SILVER/PENDING/no-Gold.

### 8. Exécuter le smoke test

Laisser `RUN_SMOKE_TEST_FIRST=True`. Commande effective:

```bash
python scripts/train_xlmr_multitask.py \
  --config configs/xlmr_multitask_colab_v0.1.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1 \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1 \
  --training-mode SCIENTIFIC_EVALUATION \
  --model-name FacebookAI/xlm-roberta-base \
  --smoke-test \
  --max-train-records 32 --max-validation-records 16 \
  --epochs 1 --max-length 128 --seed 20260718 \
  --per-device-batch-size <batch-résolu> \
  --gradient-accumulation-steps <accumulation-résolue> \
  --no-resume
```

Le CLI ajoute automatiquement `smoke-test/` aux racines:

```text
.../checkpoints/xlmr-base-multitask-v0.1/smoke-test/
.../reports/xlmr-base-multitask-v0.1/smoke-test/
```

Un smoke complet et checksummé avec resume probe PASS est réutilisé. Un état
partiel est signalé, jamais supprimé silencieusement.

### 9. Activer l’entraînement complet

Après succès du smoke test:

```python
START_FULL_TRAINING = True
```

Commande complète:

```bash
python scripts/train_xlmr_multitask.py \
  --config configs/xlmr_multitask_colab_v0.1.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1 \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1 \
  --training-mode SCIENTIFIC_EVALUATION \
  --model-name FacebookAI/xlm-roberta-base \
  --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.10 \
  --early-stopping-patience 2 --seed 20260718 \
  --epochs 4 --max-length 512 \
  --per-device-batch-size <batch-résolu> \
  --gradient-accumulation-steps <accumulation-résolue> \
  --resume
```

Le CLI conserve sérialisation contexte complet, neuf têtes, troncature par
sections, pertes multi-tâches, sélection validation, tests officiels, analyses
contrefactuelles/hard-negative/langue et checkpoints Drive atomiques.

### 10. Reprendre après déconnexion

Rouvrir depuis GitHub, choisir un GPU, conserver la même configuration et
exécuter `Run all`. Le clone est remis au même `GITHUB_REF`, l’archive locale
est revalidée/réextraite, puis `--resume` charge le dernier checkpoint complet
compatible.

Manifeste, checksums, dataset, configuration, mappings, tokens spéciaux et type
de run doivent correspondre. Optimiseur, scheduler, scaler, RNG, époque, step,
meilleur score et early stopping sont restaurés. Aucun redémarrage silencieux.

### 11. Inspecter les sorties

```text
/content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1/
/content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.1/best_model/
/content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1/
```

Les cellules finales affichent validation, neuf têtes, OOD, contrefactuels,
négatifs difficiles, comparaison EN/FR, runtime, mémoire GPU, reprise, OOM,
inventaire/checksums du modèle et commande exacte de reproduction.

## Sécurité et limites

- Aucun API key, service OpenAI, Codex, Ollama, W&B ou MLflow n’est requis.
- Aucun token Hugging Face n’est écrit dans le dépôt ou le notebook.
- Le dépôt complet est cloné depuis GitHub dans `/content`, jamais dans Drive.
- Le dataset reste SILVER; aucune promotion Gold n’est effectuée.
- Aucun split aléatoire n’est créé.
- `CLEAN_LOCAL_RUNTIME` est désactivé et ne cible jamais Drive par défaut.
