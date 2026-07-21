# PromptSec-FM XLM-R multitâche v0.2

Cette procédure crée des expériences isolées de v0.1. Elle ne modifie ni la taxonomie
PromptSec-FM v1.0, ni les 6 000 records PolicyBench, et n’utilise que le split officiel `train`
pour l’optimisation. Validation sert à l’arrêt précoce, aux seuils, au décodage du verdict et à la
sélection. Les tests, dont le français OOD, ne sont consultés qu’après sélection.

## Expériences

- `balanced` : sampling par catégorie borné, poids de classes tronqués, têtes multilabel à poids
  modestement renforcés et seuils par label; pas de perte relationnelle.
- `relational` : ajoute les batches pair-aware, la perte contrefactuelle et la cohérence du verdict;
  c’est le smoke par défaut.
- `focal` : variante relationnelle avec asymmetric focal BCE; désactivée par défaut.

Le notebook conserve `START_V0_2_TRAINING = False`, exécute le smoke B quand demandé, prépare A
et B, et laisse C désactivée. Le mode `BILINGUAL_FINAL_SILVER_MODEL` est également désactivé et ne
peut pas soutenir une affirmation OOD française non biaisée.

## Objectifs

Pour une paire invariante, chaque tête minimise la divergence de Jensen-Shannon
`JS(p_a, p_b)`. Pour une vérité modifiée, elle minimise
`max(0, margin - JS(p_a, p_b))`. Les têtes multilabel calculent ces termes séparément pour chaque
label changé ou invariant; un vecteur vide reste valide. Les deux records conservent leur perte
supervisée.

La probabilité structurée gelée est :

```text
p_detected = p(INSTRUCTION_PRESENT) × p(OPERATIVE) × p(MODEL_OR_AGENT)
             × (p(OUTSIDE_AUTHORITY) + p(SPOOFED))
```

La BCE de cohérence ne s’applique qu’aux exemples déterminés. `UNCERTAIN` n’est jamais assimilé à
`NOT_DETECTED`. Trois décodages sont conservés : `DIRECT_HEAD`,
`DERIVED_FROM_COMPONENT_HEADS` et `VALIDATION_CALIBRATED_HYBRID`. Pour l’hybride,
`p_final = alpha × p_direct + (1-alpha) × p_derived`; `alpha` est choisi sur validation avec
`verdict_macro_f1 - 0.25 × hard_negative_fpr`.

Le score robuste de checkpoint est défini avant entraînement :

```text
renormalized(0.50 × core_macro_f1
           + 0.20 × verdict_macro_f1
           + 0.15 × validation_counterfactual_sensitivity
           + 0.15 × multilabel_macro_f1)
- 0.25 × validation_hard_negative_fpr
```

Une composante indisponible est omise puis les poids positifs sont renormalisés. Aucun résultat de
test n’entre dans ce score; la perte de validation départage les ex æquo.

## Seuils et déséquilibre

La CE pondérée reste le défaut. La focal CE est optionnelle et protégée contre une combinaison de
poids extrêmes. La BCE pondérée reste le défaut multilabel; asymmetric focal BCE est stable dans
l’espace des logits et normalisée par la taille du vocabulaire, y compris pour les vecteurs tous
zéro. Les seuils suivent une grille déterministe de 0,10 à 0,90 par pas de 0,05 sur validation. Un
label sans support positif/négatif suffisant reprend 0,5, et un seuil prédisant tout positif est
écarté.

## Smoke et checkpoints

Le smoke écrit uniquement sous le sous-répertoire v0.2 `smoke-test`, démarre avec `--no-resume`
et `--reset-smoke-test`, et inclut le commit source dans la compatibilité. Une reprise issue d’un
autre commit, dataset, mapping, modèle, expérience ou configuration échoue. Le run complet utilise
`--resume`.

La rétention conserve le meilleur checkpoint et les deux derniers checkpoints complets. Avant
toute suppression, `best_model`, ses checksums, son tokenizer et ses mappings sont vérifiés. Le
mode par défaut est dry-run et écrit `checkpoint_pruning_manifest.json`. Tout chemin v0.1 est refusé.

## Commandes Colab

```bash
# Smoke relationnel propre
python scripts/train_xlmr_multitask.py --config configs/xlmr_multitask_colab_v0.2.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.2-relational \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.2-relational \
  --experiment relational --smoke-test --reset-smoke-test --no-resume \
  --max-train-records 32 --max-validation-records 16 --epochs 1 --max-length 128

# Experiment A
python scripts/train_xlmr_multitask.py --config configs/xlmr_multitask_colab_v0.2.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.2-balanced \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.2-balanced \
  --experiment balanced --resume

# Experiment B
python scripts/train_xlmr_multitask.py --config configs/xlmr_multitask_colab_v0.2.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.2-relational \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.2-relational \
  --experiment relational --resume

# Experiment C optionnel
python scripts/train_xlmr_multitask.py --config configs/xlmr_multitask_colab_v0.2.yaml \
  --dataset /content/promptsec_data/policybench-codex-v0.1 \
  --output /content/drive/MyDrive/PromptSec-FM/checkpoints/xlmr-base-multitask-v0.2-focal \
  --reports /content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.2-focal \
  --experiment focal --resume
```

Les comparaisons v0.1 lisent uniquement les rapports machine depuis
`/content/drive/MyDrive/PromptSec-FM/reports/xlmr-base-multitask-v0.1`; les poids v0.1 ne sont pas
requis. Les rapports n’impriment jamais les payloads, seulement des IDs et des hashes.
