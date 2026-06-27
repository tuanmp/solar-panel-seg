# Solar Panel Panoptic Segmentation — Design Document

## 1. Project Goal

Build a full-stack ML demonstrator for panoptic segmentation of solar photovoltaic (PV) panels from satellite/aerial imagery. The project covers the entire lifecycle: data acquisition (API pipelines), dataset curation, model fine-tuning (Mask2Former with Swin-B backbone), evaluation, experiment tracking, and production-readiness scaffolding.

Target audience: Full-stack ML / MLOps roles. Demonstrates data engineering, modeling, experimentation, and deployment-readiness skills.

## 2. Data Sources

### 2.1 Primary: BDAPPV (Kasmi et al., 2023)

- **Paper**: "A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata" — Nature Scientific Data, Jan 2023
- **License**: CC-BY 4.0
- **Content**:
  - 28,807 images from Google Earth (GSD 0.1 m/pixel, 400×400 px)
  - 17,325 images from IGN (French mapping agency, GSD 0.2 m/pixel, 400×400 px)
  - 13,303 segmentation masks (Google), 7,686 segmentation masks (IGN)
  - Installation metadata: capacity, surface, tilt, azimuth angle
- **Access**: Zenodo (https://zenodo.org/record/7358126)
- **Coverage**: France / Western Europe (rooftop residential PV)
- **Format**: .png images + .png mask rasters + metadata.csv

### 2.2 Supplemental: Bradbury et al. (2016)

- **Paper**: "Distributed solar photovoltaic array location and extent dataset for remote sensing object identification" — Nature Scientific Data, Dec 2016
- **License**: CC-BY 4.0
- **Content**:
  - 19,863 solar panel annotations across 601 images (5000×5000 px, 30 cm resolution)
  - 4 California cities: Fresno, Stockton, Oxnard, Modesto
  - Polygon vertices + centroid lat/lon + Jaccard confidence scores
- **Access**: Figshare (http://dx.doi.org/10.6084/m9.figshare.c.3255643)
- **Coverage**: California, USA (residential rooftop PV)
- **Format**: .json, .geojson, .csv, .mat + USGS orthoimagery .tif

### 2.3 Auxiliary Metadata: DeepSolar++ (Wang et al., 2022)

- **Paper**: "DeepSolar++: Understanding residential solar adoption trajectories with computer vision and technology diffusion models" — Joule, Nov 2022
- **License**: Apache-2.0
- **Content**: Census-block-group-level CSV of cumulative residential PV installations (2005–2017) across contiguous US + 420-county subset
- **Access**: S3 / GitHub (wangzhecheng/DeepSolar_timelapse)
- **Role**: Metadata enrichment for EDA notebooks; not used as training data (no segmentation masks available)

### 2.4 Dataset Merging Strategy

| Source | Images | Masks | Input Size | Preprocessing Needed |
|--------|--------|-------|-----------|---------------------|
| BDAPPV Google | 28,807 | 13,303 | 400×400 | None (already tile-sized) |
| BDAPPV IGN | 17,325 | 7,686 | 400×400 | None |
| Bradbury CA | 601 | ~19,863 | 5000×5000 | Tile 400×400 crops around centroids |

- **Total usable**: ~21K masked images (BDAPPV) + ~20K masked instances (Bradbury tiles)
- **Split**: Geographic (by region/city) to test generalization
- **Cross-provider test**: IGN images serve as distribution-shift test for models trained on Google images (and vice versa)

## 3. Data Pipeline

### 3.1 Satellite API Acquisition

Demonstrates data engineering skills with real API calls to acquire imagery on demand.

```
src/solar_seg/data/acquisition/
├── __init__.py
├── gee_client.py       # Google Earth Engine Python API client
├── sentinel_client.py  # Sentinel Hub OAuth REST API client
├── geo_utils.py        # Coordinate transforms, CRS handling, bbox helpers
└── cli.py              # CLI entry: python -m solar_seg.data.acquisition ...
```

**GEE Client** (`gee_client.py`):
- Authentication via Earth Engine service account (JSON key)
- Supports NAIP (1 m, US coverage) and Sentinel-2 (10 m, global coverage)
- Export tiles at user-specified GSD, centered on lat/lon or bounding box
- Output: GeoTIFF + metadata JSON (timestamp, CRS, cloud cover, source)

**Sentinel Hub Client** (`sentinel_client.py`):
- OAuth2 client-credentials flow
- WMS/WCS requests for Sentinel-2 L2A imagery
- Cloud coverage filter (default: < 20%)
- Date range querying for temporal analysis support
- Output: GeoTIFF + metadata JSON

**CLI** (`cli.py`):
```
python -m solar_seg.data.acquisition \
    --source gee --provider naip \
    --bbox -122.4194,37.7749,-122.4184,37.7759 \
    --output data/acquired/
```

### 3.2 Preprocessing Pipeline

```
src/solar_seg/data/preprocessing/
├── __init__.py
├── mask_converter.py    # Polygon coordinates → binary/instance rasters
├── tile_extractor.py    # Large image → fixed-size tiles around annotations
├── dataset.py           # Torch Dataset + LightningDataModule (composite)
└── augmentations.py     # Albumentations augmentation pipeline
```

**mask_converter.py**:
- BDAPPV: Read .png mask → label image where each panel instance = unique integer ID
- Bradbury: GeoJSON polygon vertices → rasterize to binary mask → connected components → instance labels
- Output: `{image_id}_semantic.png` (0=bg, 1=panel), `{image_id}_instance.png` (0=bg, 1..N=instances)

**tile_extractor.py**:
- Bradbury 5000×5000 images → 400×400 crops centered on each annotation centroid
- Discard crops with fewer than 5% panel pixels (to avoid learning background-only tiles)
- Output: same directory structure as BDAPPV for unified loading

**dataset.py**:
- `SolarSegDataset`: loads (image, semantic_mask, instance_mask) tuples
- `SolarSegDataModule`: LightningDataModule with train/val/test splits
- Handles mixed-resolution inputs (BDAPPV at 0.1-0.2m GSD, Bradbury at 0.3m GSD) via resize to uniform 400×400
- Supports optional metadata loading (capacity, tilt, azimuth)

**augmentations.py** (Albumentations):
- Training: RandomCrop(384,384), HorizontalFlip(p=0.5), VerticalFlip(p=0.1), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ElasticTransform
- Validation/Test: Resize(400,400) only
- All augmentations applied simultaneously to image and masks for consistency

### 3.3 Data Versioning (DVC)

- DVC tracks raw data, preprocessed data, and model checkpoints
- Pipeline stages:
  1. `download_bdappv.dvc` → downloads from Zenodo
  2. `download_bradbury.dvc` → downloads from Figshare
  3. `preprocess.dvc` → runs mask conversion + tiling
  4. `train.dvc` → training run
- Data stored on local filesystem (HPC scratch) or optional S3 remote
- `dvc repro` for end-to-end reproducibility

## 4. Model Architecture

### 4.1 Mask2Former

Framework: **Mask2Former** (Cheng et al., 2022, CVPR) — unified architecture for semantic, instance, and panoptic segmentation.

Components:
- **Backbone**: Swin-B (Swin Transformer Base, window attention, 88M params)
  - Pre-trained on ImageNet-22K
  - Output multi-scale feature maps (1/4, 1/8, 1/16, 1/32 resolution)
- **Pixel Decoder**: Multi-scale deformable attention transformer
  - Fuses features from all backbone stages
  - Outputs per-pixel embeddings at 1/4 resolution
- **Transformer Decoder**: 6-layer transformer with masked attention
  - Learns N = 100 object queries
  - Each query predicts: binary mask logits + class logits + bounding box
  - Masked attention restricts cross-attention to predicted mask region (efficient, reduces false positives)
- **Loss**:
  - Binary mask loss: sigmoid cross-entropy + Dice loss (per-query)
  - Class loss: cross-entropy
  - Bipartite matching via Hungarian algorithm (assigns queries to ground-truth instances)
  - No aux loss weighting needed beyond default (CE: 2.0, Dice: 5.0)

Outputs:
- **Panoptic**: Class ID + instance ID per pixel (stuff classes merged)
- **Semantic**: Max-probability class per pixel
- **Instance**: Each mask with class + confidence

Implementation: HuggingFace `transformers` (`Mask2FormerForUniversalSegmentation`). Integrated as a LightningModule in `src/solar_seg/models/mask2former_module.py`.

### 4.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Input resolution | 384×384 (random crop from 400×400) |
| Backbone | Swin-B (ImageNet-22K pre-trained) |
| Object queries | 100 |
| Batch size | 8 per GPU (scales with HPC multi-GPU) |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.05) |
| Scheduler | Poly learning rate (power=1.0) |
| Training epochs | 50 |
| Warmup | Linear warmup (1,000 steps), lr=1e-6 → 1e-4 |
| Mixed precision | bfloat16 (A100/H100 native) |
| Gradient clipping | max_norm=0.1 |

## 5. Training & Experimentation

### 5.1 Training Orchestration

- **Lightning** as trainer harness (retained from existing template)
- **Hydra** for config management:
  - `configs/data/bdappv.yaml`, `configs/data/bradbury.yaml` → dataset-specific
  - `configs/model/mask2former_swin_b.yaml` → model config
  - `configs/trainer/base.yaml` → trainer config (epochs, optimizer, scheduler)
  - Composition: `python train.py --config-dir configs --config-name experiment data=bdappv model=mask2former_swin_b`
  - CLI overrides: `data.batch_size=16 trainer.max_epochs=100`

### 5.2 Experiment Tracking (MLflow)

- **What**: Hyperparameters, configs, metrics (PQ, SQ, RQ, IoU, mIoU, loss curves), model checkpoints, visualizations
- **Where**: `mlruns/` directory (optionally S3 remote for sharing)
- **Per-run**: Auto-log config snapshot, metrics per epoch, best checkpoint
- **Comparison**: MLflow UI to compare runs across dataset variants, hyperparameters, backbones

### 5.3 Metrics

Primary:
- **Panoptic Quality (PQ)** = Σ(pred ∩ gt) / Σ(pred ∪ gt) averaged over instances
- Semantic Quality (SQ) — average IoU of matched pairs
- Recognition Quality (RQ) — F1 score of instance detection

Secondary:
- **Mean IoU** (semantic segmentation)
- **Boundary IoU** (panel edge accuracy)
- **F1 score** per image (instance detection)
- **Inference speed** (FPS on one A100)

## 6. Evaluation & Ablation Studies

Ablation dimensions:
1. **Backbone scaling**: Swin-T (tiny, cheaper) vs Swin-B (base) vs Swin-L (large, most expensive)
2. **Dataset composition**: BDAPPV-only vs BDAPPV + Bradbury
3. **Cross-provider shift**: Train on Google → eval on IGN and vice versa (distribution shift robustness)
4. **Augmentation**: With vs without photometric augmentation
5. **Loss weighting**: Default vs tuned weight for Dice vs CE

### 6.1 Package Transition

The existing codebase uses `src/ml_cookbook/` as the Python package. The solar panel project builds a new `src/solar_seg/` package alongside it. The existing `ml_cookbook` package is preserved for reference but all new development lives under `solar_seg`. The root `train.py` entrypoint is updated to delegate to `solar_seg` training.

## 7. Project Structure

```
solar-panel-seg/
├── docs/
│   └── superpowers/specs/
│       └── 2026-06-27-solar-panel-segmentation-design.md
├── configs/
│   ├── data/
│   │   ├── bdappv.yaml
│   │   └── bradbury.yaml
│   ├── model/
│   │   ├── mask2former_swin_t.yaml
│   │   ├── mask2former_swin_b.yaml
│   │   └── mask2former_swin_l.yaml
│   ├── trainer/
│   │   └── base.yaml
│   └── experiment/
│       ├── bdappv_only.yaml
│       ├── combined.yaml
│       └── cross_provider.yaml
├── data/                      # DVC-tracked (git-ignored)
│   ├── raw/                   # Original downloads
│   │   ├── bdappv/            # Zenodo extraction
│   │   └── bradbury/          # Figshare extraction
│   └── processed/             # Preprocessed masks + tiles
├── src/
│   ├── __init__.py
│   ├── train.py               # Entry point (Hydra + Lightning training)
│   ├── solar_seg/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── acquisition/   # API clients for satellite imagery
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gee_client.py
│   │   │   │   ├── sentinel_client.py
│   │   │   │   ├── geo_utils.py
│   │   │   │   └── cli.py
│   │   │   └── preprocessing/
│   │   │       ├── __init__.py
│   │   │       ├── dataset.py
│   │   │       ├── mask_converter.py
│   │   │       ├── tile_extractor.py
│   │   │       └── augmentations.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── mask2former_module.py   # LightningModule wrapping Mask2Former
│   │   │   └── backbones.py            # Swin backbone configs
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── lightning_module.py     # Shared training/val/test step hooks
│   │   │   └── callbacks.py            # Custom Lightning callbacks
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py              # PQ, SQ, RQ, IoU calculators
│   │   │   ├── visualization.py        # Overlay masks, side-by-side panels
│   │   │   └── ablations.py            # Ablation study runner
│   │   └── utils/
│   │       ├── __init__.py
│   └──       └── repro.py              # Seed + determinism (existing)
├── notebooks/
│   ├── 01_eda.ipynb                    # Dataset exploration & statistics
│   ├── 02_results.ipynb                # Results analysis & visualization
│   └── 03_ablations.ipynb              # Ablation study report
├── tests/
│   ├── test_shapes.py                  # Existing shape tests (update)
│   ├── test_train_smoke.py             # Existing smoke test (update)
│   ├── test_data/
│   │   ├── test_mask_converter.py
│   │   └── test_tile_extractor.py
│   ├── test_models/
│   │   └── test_mask2former_shapes.py
│   └── test_acquisition/
│       ├── test_gee_client.py
│       └── test_sentinel_client.py
├── pyproject.toml
├── Makefile
├── setup.cfg
└── train.py                    # Root entry point (forward to src/train.py)
```

## 8. Deployment (Future — Separate Repo)

Considerations for downstream deployment repo:
- **Model export**: ONNX → TensorRT or TorchScript
- **Serving**: Triton Inference Server or FastAPI + ONNX Runtime
- **Containerization**: Docker multi-stage build (base image + model weights)
- **Inference pipeline**: Accept GeoTIFF/URL → tile → segment → merge → output GeoJSON with panel polygons
- **CI/CD**: Automated build + test + deploy on push
- **Monitoring**: Inference latency, throughput, drift detection on predicted outputs

## 9. Success Criteria

1. **Training pipeline runs end-to-end** on HPC with Mask2Former + BDAPPV dataset
2. **PQ ≥ 45** on BDAPPV test split (reasonable benchmark target for panoptic solar)
3. **API pipelines** (GEE + Sentinel Hub) query and download real imagery successfully
4. **Ablation notebook** documents cross-provider distribution shift results
5. **All tests pass** (`make test`)
6. **DVC pipeline** fully reproducible (`dvc repro`)
7. **MLflow UI** shows experiment history with comparison views

## 10. Dependencies

Current (`pyproject.toml`): torch, lightning, torchvision, numpy, pyyaml

Add:
- `transformers>=4.38` (HuggingFace Mask2Former)
- `albumentations>=1.3` (augmentation)
- `hydra-core>=1.3` (config management)
- `mlflow>=2.10` (experiment tracking)
- `dvc>=3.0` (data versioning)
- `earthengine-api>=1.0` (GEE Python client)
- `sentinelhub>=3.10` (Sentinel Hub Python client)
- `rasterio>=1.3` (GeoTIFF handling)
- `opencv-python>=4.8` (polygon-to-mask rasterization)
- `scikit-image>=0.22` (connected components)
- `matplotlib>=3.7` (visualization)
