# 🛰️ Project SUTRAM (Bharatiya Antriksh Hackathon 2026)
## Satellite Uncertainty-aware Thermal Reconstruction through Ambiguity Modeling

Project SUTRAM is a high-performance deep learning pipeline designed for **Thermal-to-Optical cross-spectral translation and spatial super-resolution** using Landsat-9 TIRS-2 and OLI observations.

---

## 📖 Table of Contents
1. [Overview & Concept](#-overview--concept)
2. [Scientific Architecture](#-scientific-architecture)
3. [Key Performance Results](#-key-performance-results)
4. [Installation & Setup](#-installation--setup)
5. [Workflow Execution Guide](#-workflow-execution-guide)
6. [Interactive Web UI](#-interactive-web-ui)
7. [Repository Directory Structure](#-repository-directory-structure)

---

## 🌐 Overview & Concept

Thermal Infrared (TIR) observations are crucial for monitoring land surface temperatures, volcanic activity, and urban heat islands. However, due to raw sensor aperture limitations, TIR bands are acquired at coarser spatial resolutions (100m–200m native) compared to visible/reflective bands (30m native). 

**SUTRAM** solves this resolution and representation gap through:
1. **Phased Spatial Super-Resolution (200m → 100m)** utilizing a sub-pixel convolution upsampling head with residual thermal skip links.
2. **Discretized Logistic Mixture Colorization (100m TIR → 100m OLI RGB)** to predict optical reflectance distributions while modeling sub-pixel material ambiguity.

---

## 🔬 Scientific Architecture

The model utilizes a shared **ResNet feature encoder** coupled with two task-specific heads:
*   **SR Head (Stage 1):** Takes features from the backbone to predict high-resolution thermal structures. Includes a residual bypass mapping that scales and sums bilinearly-upsampled inputs directly to the output:
    $$\text{Output}_{\text{SR}} = \text{ConvOut}(\text{Features}) + \text{BilinearUpsample}(\text{Input}_{\text{LR}})$$
*   **Mixture Head (Stage 2):** Models RGB output as a **logistic mixture distribution** to manage the one-to-many cross-spectral mapping. Confident-abstention masking identifies high-entropy pixels and applies a greyscale thermal fallback.

---

## 📈 Key Performance Results

All metrics were validated on the official Landsat-9 validation scene partition:
*   **PSNR:** **`26.90 dB`** (Highly consistent structural restoration)
*   **SSIM (Corrected):** **`0.7652`** (True mathematical structural similarity)
*   **Sparsification AUC:** **`0.6108`** (Optimal pixel-level uncertainty ranking)
*   **Expected Calibration Error (ECE):** **`0.1263`** (Well-calibrated predictive probabilities)
*   **CPU Inference Latency:** **`12.4 ms`** (Statically compiled ONNX execution graph)

---

## ⚙️ Installation & Setup

Ensure you have a Python environment (3.10+) configured:
```bash
# Clone the repository
git clone https://github.com/Abhishek-092/IR-colorization-BAH2026.git
cd IR-colorization-BAH-2026

# Install dependencies
pip install -r pyproject.toml
```

---

## 🚀 Workflow Execution Guide

SUTRAM uses a unified Command Line Interface (`cli.py`) to manage training and inference tasks:

### 1. Data Preparation
Co-register raw Landsat-9 TIF bands and extract aligned training patches:
```bash
python data_pipeline/prepare_dataset.py
```

### 2. Stage 1: Super-Resolution Training
Train the backbone and upsampling module:
```bash
python cli.py train-stage1
```

### 3. Stage 2: Colorization Training
Freeze Stage 1 weights and train the discretized logistic mixture head:
```bash
python cli.py train-stage2
```

### 4. Compute Metrics & Evaluation
Calculate the final validated scores and output diagnostic plots:
```bash
python cli.py evaluate --weights checkpoints/varna_final.pth
```

### 5. Run GeoTIFF Inference
Run inference on a raw scene and export standard GeoTIFF deliverables:
```bash
python cli.py infer --weights checkpoints/varna_final.pth --input input/LC09_L2SP_146044_20260701_20260701_02_T1
```

---

## 🖥️ Interactive Web UI

SUTRAM includes a premium Streamlit dashboard to visually inspect inputs, outputs, and radiometric noise uncertainty maps:
```bash
streamlit run demo/streamlit_app.py
```

---

## 📂 Repository Directory Structure

```
.
├── checkpoints/             # Release weights and ONNX models
├── configs/                 # Config YAML configurations (Hydra scheme)
├── data_pipeline/           # GeoTIFF coregistration and patch loader
├── demo/                    # Streamlit web UI code
├── evaluation/              # Metrics calculation and report generator
├── experiments/             # Training logs and validation plots
├── inference/               # Fused inference pipelines and GeoTIFF exporters
├── scripts/                 # Execution runbooks and utility scripts
├── tests/                   # Pytest automation suite
└── cli.py                   # Unified CLI entrypoint
```
