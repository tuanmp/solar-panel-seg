# Solar Panel Segmentation — Experiment Log

## Completed Runs — Phase 1: Hyperparameter Tuning

### E1 — Baseline Swin-B v1 (old config)
| | |
|---|---|
| MLflow run | `8eb57a4204304d7b9067bc0baca52845` |
| Model | Swin-B (47M) |
| LR / wd | 1e-4 / 0.05 |
| Warmup | 1000 steps |
| Batch size | 8 |
| Augmentations | RandomCrop 384, HFlip 0.5, VFlip 0.1, ColorJitter 0.8, GaussBlur 0.1 |
| Epochs | 14 |
| **test/loss** | **8.51** |
| test/dice | 0.662 |
| test/ce | 0.074 |

### E2 — Tuned Swin-B v2 (improved config)
| | |
|---|---|
| MLflow run | `2cb9b6ce92b447f88c47902c81f93243` |
| LR / wd | **3e-5** / **0.01** |
| Warmup | **500** steps |
| Batch size | **16** |
| Augmentations | No crop, **HFlip 0.3 only** |
| Epochs | 11 |
| **test/loss** | **8.07** (−5.2% vs E1) |
| test/dice | 0.639 |
| test/ce | 0.068 |

### E6 — Baseline Swin-B v2 Reproduce (job 55175516)
| | |
|---|---|
| Slurm ID | 55175516 |
| Epochs | 17 (completed via requeue) |
| **test/loss** | **8.15** |
| test/dice | 0.633 |
| test/ce | **0.063** (best so far) |

### E3 — Swin-T (job 55175504)
| | |
|---|---|
| Slurm ID | 55175504 |
| Model | Swin-T (47M) |
| Epochs | 17 |
| **test/loss** | **8.27** |
| test/dice | 0.637 |
| test/ce | 0.080 |
| Verdict | Swin-B beats Swin-T on BDAPPV |

### E5 — No Horizontal Flip (job 55175509)
| | |
|---|---|
| Slurm ID | 55175509 |
| Augmentations | **HFlip 0.0**, ColorJitter 0.8, GaussBlur 0.1 |
| **test/loss** | **8.15** |
| test/dice | **0.627** (best dice!) |
| test/ce | 0.076 |
| Verdict | HFlip negligible ~0.1 loss difference |

### E4 — Lower LR 1e-5 on Swin-B (job 55175505)
| | |
|---|---|
| Slurm ID | 55175505 |
| LR | **1e-5** |
| **test/loss** | **8.60** |
| test/dice | 0.666 |
| test/ce | 0.077 |
| Verdict | 3e-5 is the sweet spot |

---

## Running / Pending — Phase 2: Ceiling Breakers

### E7 — Freeze Backbone (code change, resubmitted)
| | |
|---|---|
| Slurm ID | 55188551 |
| Branch | `feat/freeze-backbone` |
| Clone | `/pscratch/sd/p/pmtuan/solar-exp-freeze` |
| Config | `model=mask2former_swin_b_frozen` |

### E8 — EMA Weight Averaging (code change, resubmitted)
| | |
|---|---|
| Slurm ID | 55188552 |
| Branch | `feat/ema` |
| Clone | `/pscratch/sd/p/pmtuan/solar-exp-ema` |
| Config | `trainer.ema_decay=0.999` |

### E9 — 50-Epoch Baseline (preempt QOS)
| | |
|---|---|
| Slurm ID | 55188935 |
| QOS | preempt (0.25× after 2h) |
| Config | default, 50 epochs |
| Hypothesis | 18 epochs may not be enough for convergence |

### E10 — BDAPPV + Bradbury Combined + Scale Jitter
| | |
|---|---|
| Slurm ID | 55188991 |
| Data | `data=combined` (21K + 14K tiles) |
| Aug | Scale jitter 0.2 |
| Hypothesis | More diverse data → break overfitting ceiling |

### E11 — Scale Jitter Only (BDAPPV)
| | |
|---|---|
| Slurm ID | 55188992 |
| Data | `data=bdappv_scale_jitter` |
| Aug | Scale jitter 0.2, no Bradbury |
| Hypothesis | Multi-scale alone helps regularization |

### E12 — Reweighted Loss (dice 10 : mask 5 : ce 1)
| | |
|---|---|
| Slurm ID | 55188993 |
| Model | `model=mask2former_swin_b_reweighted` |
| Weights | ce=1.0, mask=5.0, dice=10.0 |
| Hypothesis | More dice pressure → better shapes |

### E13 — Swin-L (scaling test)
| | |
|---|---|
| Slurm ID | 55189001 |
| Model | `model=mask2former_swin_l` |
| Hypothesis | More capacity with current regularization |

---

## Comparison Matrix

| Exp | test/loss | test/dice | test/ce | Key finding |
|-----|-----------|-----------|---------|-------------|
| E1 (old baseline) | 8.51 | 0.662 | 0.074 | Aggressive augs |
| E2 (v2 baseline) | 8.07 | 0.639 | 0.068 | Tuning works |
| **E6 (v2 rep)** | **8.15** | 0.633 | **0.063** | Best CE, reproducible |
| E3 (Swin-T) | 8.27 | 0.637 | 0.080 | Bigger > smaller |
| E5 (no flip) | 8.15 | **0.627** | 0.076 | Best dice, HFlip negligible |
| E4 (LR 1e-5) | 8.60 | 0.666 | 0.077 | Underfitting |
| E7 (freeze) | ? | ? | ? | Pending |
| E8 (EMA) | ? | ? | ? | Pending |
| E9 (50ep) | ? | ? | ? | Pending |
| E10 (combined+scale) | ? | ? | ? | Pending |
| E11 (scale jitter) | ? | ? | ? | Pending |
| E12 (reweighted) | ? | ? | ? | Pending |
| E13 (Swin-L) | ? | ? | ? | Pending |

## Key Insight

**We hit a ceiling at test/loss ≈ 8.1 on BDAPPV alone.** Every config that converges lands between 8.07–8.27 despite different augmentations, backbones, and learning rates. The train loss continues to ~5.0 while validation stalls. This is structural — more hyperparameter tuning won't help.

**Phase 2 strategy:** break through with more data (Bradbury), scale jitter (multi-scale augmentation), loss rebalancing, and longer training. The code-change experiments (freeze, EMA) test architectural regularization.
