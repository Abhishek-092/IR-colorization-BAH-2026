# Train SUTRAM on a Kaggle GPU (≈10–15 min instead of ~5 h)

Your Mac's MPS runs the 512² mixture head at ~15–30 min/epoch. A free Kaggle
GPU (T4 ×2 / P100) does it in well under a minute/epoch. Here's the whole flow.

## What you upload
`sutram_kaggle_bundle.zip` (1.7 GB) — already built in the repo root. It contains
**all the code + the raw Landsat scenes** (no pre-made patches, so the upload
stays small; patches are regenerated on Kaggle in a couple of minutes).

## Steps

1. **Create a Kaggle account** (free) and verify your phone number — this is
   required to enable GPU + internet in notebooks.

2. **Upload the bundle as a Dataset**
   - Kaggle → *Create* → *New Dataset*
   - Drag in `sutram_kaggle_bundle.zip` (Kaggle auto-unzips it)
   - Title it e.g. `sutram-bundle`, create.

3. **New Notebook**
   - Kaggle → *Create* → *New Notebook*
   - Right panel → *Add Input* → your `sutram-bundle` dataset
   - Right panel → *Settings*:
     - **Accelerator: GPU T4 ×2** (or P100)
     - **Internet: On** (needed to pip-install rasterio)

4. **Paste the training script**
   - Copy the entire contents of `kaggle/sutram_kaggle_train.py` into the first
     notebook cell.
   - *Run All*.

5. **Wait ~10–15 min.** The script:
   - regenerates colour + SR patches from the raw scenes,
   - trains Stage 1 (super-resolution) and Stage 2 (colour) on the GPU,
   - evaluates and packages the checkpoints.

6. **Download the results**
   - When it finishes, the last cell prints the files under
     `/kaggle/working/sutram_trained/`.
   - Right panel → *Output* → download `sutram_trained/` (or the individual
     `.pth` files): `backbone_stage1.pth`, `sr_head_stage1.pth`,
     `mixture_head_stage2.pth`, `sutram_final.pth`, `metrics.json`.

7. **Bring them back here**
   - Drop the downloaded `.pth` files into
     `experiments/sutram_baseline/checkpoints/` (and `checkpoints/` for
     `sutram_final.pth`) in this repo.
   - Tell me "checkpoints are back" and I'll repackage, restart the dashboard,
     render the reconstruction, and commit.

## Notes
- The bundle forces `device: "cuda"` and bumps batch size to 32 on Kaggle; your
  local configs are untouched.
- If Kaggle's GPU quota is exhausted (30 h/week free), wait for the weekly reset
  or use the "Save Version → Save & Run All (Commit)" background run.
- Google Colab works too with the same script — just adjust the input path
  (`/content/...` instead of `/kaggle/input/...`).
