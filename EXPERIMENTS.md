# Solar Panel Segmentation — Experiment Log

## Completed Runs

### E0 — Early Swin-T Smoke Test (Swin-T, LR 1e-4, wd 0.05)
| | |
|---|---|
| MLflow run | `d298c5365a5e4ccbb727ae0bec0d1369` |
| Date | 2026-06-27 |
| Model | Swin-T (47.4M params) |
| Batch size | 4 |
| Augmentations | RandomCrop 384, HFlip 0.5, VFlip 0.1 |
| Epochs | 1 (sanity check) |
| test/loss | 22.38 |
| Notes | Early config validation. Found `_prepare_labels` bug (fixed). |

### E1 — Baseline Swin-B v1 (Swin-B, LR 1e-4, wd 0.05)
| | |
|---|---|
| MLflow run | `8eb57a4204304d7b9067bc0baca52845` |
| Model | Swin-B (47M) |
| LR / wd | 1e-4 / 0.05 |
| Warmup | 1000 steps |
| Batch size | 8 |
| Augmentations | RandomCrop 384, HFlip 0.5, VFlip 0.1, ColorJitter 0.8, GaussBlur 0.1 |
| Epochs | 14 |
| **Best val/loss** | **8.51** (epoch 8) |
| Best val/dice | 0.657 |
| Best val/ce | 0.070 |
| **test/loss** | **8.51** |
| test/dice | 0.662 |
| test/ce | 0.074 |
| Notes | Overfitting: train-val gap grew 0.6→3.0. Val plateaued at epoch 2. |

### E2 — Tuned Swin-B v2 (Swin-B, LR 3e-5, wd 0.01)
| | |
|---|---|
| MLflow run | `2cb9b6ce92b447f88c47902c81f93243` |
| Model | Swin-B (47M) |
| LR / wd | **3e-5** / **0.01** |
| Warmup | **500** steps |
| Batch size | **16** |
| Augmentations | No crop, **HFlip 0.3 only**, ColorJitter 0.8, GaussBlur 0.1 |
| Epochs | 11 |
| **Best val/loss** | **8.06** (epoch 6, **+5.3% vs E1**) |
| Best val/dice | 0.636 (+3.1%) |
| Best val/ce | 0.064 (+9.1%) |
| **test/loss** | **8.07** (-5.2%) |
| test/dice | 0.639 (-3.4%) |
| test/ce | 0.068 (-8.0%) |
| Notes | All metrics improved despite fewer epochs. Overfitting gap still ~3. |

---

## Running / Pending Experiments

All submitted 2026-06-28. Shared QOS, 4h walltime, ~18 epochs each.

### E3 — Swin-T (config-only)
| | |
|---|---|
| Slurm ID | `55175504` |
| Branch | `main` |
| Hydra override | `model=mask2former_swin_t` |
| MLflow name | `swin_t` |
| Model | Swin-T (30M) |
| LR / wd | 3e-5 / 0.01 (from config) |
| Batch size | 16 |
| Augmentations | No crop, HFlip 0.3 (from `bdappv`) |
| Hypothesis | Fewer params → less overfit on 21K tiles |

### E4 — Swin-B, Lower LR (config-only)
| | |
|---|---|
| Slurm ID | `55175505` |
| Branch | `main` |
| Hydra override | `model.learning_rate=1.0e-5` |
| MLflow name | `swin_b_lr1e5` |
| Model | Swin-B (47M) |
| LR / wd | **1e-5** / 0.01 |
| Hypothesis | Finer LR → deeper convergence, less overfitting |

### E5 — No Horizontal Flip (config-only)
| | |
|---|---|
| Slurm ID | `55175509` |
| Branch | `main` |
| Hydra override | `data=bdappv_noflip` |
| MLflow name | `noflip_swinb` |
| Augmentations | **HFlip 0.0**, ColorJitter 0.8, GaussBlur 0.1 |
| Hypothesis | Aerial panels have consistent orientation — flips may hurt |

### E6 — Baseline Swin-B v2 Repeat (config-only)
| | |
|---|---|
| Slurm ID | `55175516` |
| Branch | `main` |
| MLflow name | `baseline_swinb_v2` |
| Config | Default (`configs/default.yaml`) |
| Notes | Fresh run at commit `13b4432` for fair comparison |

### E7 — Freeze Backbone (code change)
| | |
|---|---|
| Slurm ID | `55175613` |
| Branch | `feat/freeze-backbone` (commit `3f1e3ce`) |
| Clone path | `/pscratch/sd/p/pmtuan/solar-exp-freeze` |
| Hydra override | `model=mask2former_swin_b_frozen` |
| MLflow name | `freeze_swinb` |
| Config | `freeze_backbone: true`, `unfreeze_epoch: 25` |
| Hypothesis | Train head first ~25 epochs, then unfreeze backbone → better generalization |

### E8 — EMA Weight Averaging (code change)
| | |
|---|---|
| Slurm ID | `55175614` |
| Branch | `feat/ema` (commit `2fdd8a5`) |
| Clone path | `/pscratch/sd/p/pmtuan/solar-exp-ema` |
| Hydra override | `trainer.ema_decay=0.999` |
| MLflow name | `ema_swinb` |
| Config | EMA decay 0.999, warmup from 0.5 |
| Hypothesis | EMA smooths training noise → better generalization |

---

## Comparison Matrix

| Exp | MLflow name | Model | LR | wd | Batch | Aug | Best val/loss | Best val/dice | test/loss |
|-----|------------|-------|-----|------|------|------|--------------|---------------|-----------|
| E1  | (8eb57a) | Swin-B | 1e-4 | 0.05 | 8 | crop+flip | 8.51 | 0.657 | 8.51 |
| E2  | (2cb9b6) | Swin-B | 3e-5 | 0.01 | 16 | hflip 0.3 | **8.06** | **0.636** | **8.07** |
| E3  | `swin_t` | Swin-T | 3e-5 | 0.01 | 16 | hflip 0.3 | ? | ? | ? |
| E4  | `swin_b_lr1e5` | Swin-B | 1e-5 | 0.01 | 16 | hflip 0.3 | ? | ? | ? |
| E5  | `noflip_swinb` | Swin-B | 3e-5 | 0.01 | 16 | **no flip** | ? | ? | ? |
| E6  | `baseline_swinb_v2` | Swin-B | 3e-5 | 0.01 | 16 | hflip 0.3 | ? | ? | ? |
| E7  | `freeze_swinb` | Swin-B | 3e-5 | 0.01 | 16 | hflip 0.3 | ? | ? | ? |
| E8  | `ema_swinb` | Swin-B | 3e-5 | 0.01 | 16 | hflip 0.3 | ? | ? | ? |

## Commands

```bash
# View queue
squeue -u pmtuan

# MLflow UI
mlflow ui --backend-store-uri /global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg/mlruns

# Clone a feature branch for local testing
git clone /global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg /pscratch/sd/p/pmtuan/solar-exp-freeze
cd /pscratch/sd/p/pmtuan/solar-exp-freeze && git checkout feat/freeze-backbone
```
