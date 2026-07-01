import os
import glob
import numpy as np
import torch
import streamlit as st
from PIL import Image

from training.backbone import ResNetBackbone
from training.sr_head import SRHead
from training.mixture_head import MixtureHead
from inference.pipeline import VARNAInferencePipeline
from evaluation.visualization import percentile_stretch

st.set_page_config(layout="wide", page_title="VARNA: TIR Colorization and Enhancement")

st.title("🛰️ Project VARNA (Bharatiya Antriksh Hackathon 2026)")
st.subheader("Variational class-Aware Radiance-to-reflectance Network")

# Sidebar configurations
st.sidebar.header("Model Configuration")
K_components = st.sidebar.slider("Mixture Components (K)", min_value=6, max_value=8, value=6)
confidence_threshold = st.sidebar.slider("Abstention Variance Threshold", 0.0, 1.0, 0.25)

# Load pipeline models
@st.cache_resource
def load_varna_pipeline(K):
    backbone = ResNetBackbone()
    sr_head = SRHead()
    mix_head = MixtureHead(K=K)
    
    # Check if weights exist, otherwise initialize randomly for demo
    checkpoint_dir = os.path.join("experiments", "varna_baseline", "checkpoints")
    bb_path = os.path.join(checkpoint_dir, "backbone_stage1.pth")
    sr_path = os.path.join(checkpoint_dir, "sr_head_stage1.pth")
    mix_path = os.path.join(checkpoint_dir, "mixture_head_stage2.pth")
    
    if os.path.exists(bb_path):
        backbone.load_state_dict(torch.load(bb_path, map_location="cpu"))
    if os.path.exists(sr_path):
        sr_head.load_state_dict(torch.load(sr_path, map_location="cpu"))
    if os.path.exists(mix_path):
        mix_head.load_state_dict(torch.load(mix_path, map_location="cpu"))
        
    pipeline = VARNAInferencePipeline(backbone, sr_head, mix_head, K=K)
    pipeline.eval()
    return pipeline

pipeline = load_varna_pipeline(K_components)

# Dataset selection
st.sidebar.header("Dataset Explorer")
patch_dirs = glob.glob(os.path.join("output", "patches", "*", "sample_*"))

if not patch_dirs:
    st.info("No patch samples found in output/patches. Please run driver.py first.")
else:
    selected_sample = st.sidebar.selectbox("Select Sample Patch", patch_dirs)
    
    # Load sample arrays
    tir_200 = np.load(os.path.join(selected_sample, "tir_200m.npy"))
    tir_100_gt = np.load(os.path.join(selected_sample, "tir_100m_512.npy"))
    rgb_100_gt = np.load(os.path.join(selected_sample, "rgb_100m_512.npy"))
    
    # Reshape RGB if channel-last
    if rgb_100_gt.ndim == 3 and rgb_100_gt.shape[0] != 3:
         rgb_100_gt = np.moveaxis(rgb_100_gt, -1, 0)
         
    # Run Inference
    input_tensor = torch.from_numpy(tir_200).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        sr_tir, decode_outs = pipeline(input_tensor)
        
    sr_np = sr_tir.squeeze().numpy()
    pred_rgb = decode_outs["dominant_color"].squeeze().numpy()
    within_var = decode_outs["within_mode_variance"].squeeze().numpy()
    between_var = decode_outs["between_mode_variance"].squeeze().numpy()
    entropy = decode_outs["entropy"].squeeze().numpy()

    # Step-by-Step Panels
    tab1, tab2, tab3 = st.tabs(["📊 Stage Outputs", "🔍 Uncertainty & Abstention", "⏱️ Technical Benchmark"])
    
    with tab1:
        st.write("### Stage 1: Spatial Super-Resolution (200m → 100m)")
        col1, col2 = st.columns(2)
        with col1:
            st.image(percentile_stretch(tir_200), caption="Input TIR (200m)", use_column_width=True)
        with col2:
            st.image(percentile_stretch(sr_np), caption="Super-Resolved TIR (100m)", use_column_width=True)

        st.write("---")
        st.write("### Stage 2: Color Synthesis and Reflectance Mapping")
        col3, col4 = st.columns(2)
        with col3:
            # Transpose to (H,W,C) for visualization
            pred_rgb_viz = np.moveaxis(pred_rgb, 0, -1)
            st.image(percentile_stretch(pred_rgb_viz), caption="VARNA Synthesized Colorized RGB (100m)", use_column_width=True)
        with col4:
            rgb_gt_viz = np.moveaxis(rgb_100_gt, 0, -1)
            st.image(percentile_stretch(rgb_gt_viz), caption="Ground-Truth OLI RGB (100m)", use_column_width=True)

    with tab2:
        st.write("### Variational Uncertainty Decomposition")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(percentile_stretch(within_var), caption="Within-mode Variance (Radiometric Noise)", use_column_width=True)
        with col2:
            st.image(percentile_stretch(between_var), caption="Between-mode Variance (Material Ambiguity)", use_column_width=True)
        with col3:
            st.image(percentile_stretch(entropy), caption="Mixing Coefficient Entropy", use_column_width=True)

        st.write("### Confident-Abstention Masking")
        # Abstain if total variance exceeds threshold
        total_var = within_var + between_var
        max_v = total_var.max() if total_var.max() > 0 else 1.0
        normalized_var = total_var / max_v
        
        abstain_mask = normalized_var > confidence_threshold
        
        col4, col5 = st.columns(2)
        with col4:
            st.image(abstain_mask.astype(np.uint8) * 255, caption="Abstention Mask (White = Abstain)", use_column_width=True)
        with col5:
            # Overwrite abstained pixels with greyscale thermal
            calibrated_thermal = percentile_stretch(sr_np)
            final_display = pred_rgb_viz.copy()
            final_display = percentile_stretch(final_display)
            for c in range(3):
                final_display[..., c] = np.where(abstain_mask, calibrated_thermal, final_display[..., c])
            st.image(final_display, caption="Final Output with Greyscale Overlay on Abstentions", use_column_width=True)

    with tab3:
        st.write("### Model Execution & Performance Profile")
        st.json({
            "Backbone parameters": "~2.5M",
            "SR Head parameters": "~0.3M",
            "Mixture Head parameters": "~0.5M",
            "Precision Mode": "Float32 (Decode Submodule)",
            "Device": "CPU (Streamlit Hosting)"
        })
