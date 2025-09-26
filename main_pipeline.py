"""
main_pipeline.py

End-to-end pipeline for anomaly detection on MVTec AD 2 dataset.
Flow:
    1. Load images
    2. Run adaptive gating (decides single vs multi-scale)
    3. Extract features (single or multi-scale)
    4. Apply anomaly core / boosting ensemble
    5. Postprocess anomaly maps
    6. Compute evaluation metrics
"""

import os
from PIL import Image
import numpy as np

# ---------- Our modules ----------
from pca_mask import pca_foreground_mask
from adaptive_gate import compute_descriptors, gbdt_gate
from feature_extract import extract_spatial_features
from anomaly_core import compute_anomaly_scores
from boosting_ensemble import boosting_refinement
from postprocess import refine_anomaly_map
from eval_metrics import evaluate_results

def run_pipeline(data_root, category, split="validation", gbdt_model_path=".src/models/gbdt_model.pkl"):
    # Set dataset paths
    img_folder = os.path.join(data_root, category, category, split)
    if not os.path.exists(img_folder):
        raise RuntimeError(f"Folder not found: {img_folder}")

    # Load all images
    image_files = [
        os.path.join(img_folder, f)
        for f in os.listdir(img_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    all_scores = []
    all_labels = []  # if labels are available
    all_masks = []   # if masks are available

    # Process each image
    for img_path in image_files:
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)

        # Step 2: PCA mask
        mask = pca_foreground_mask(np_img)

        # Step 3: Adaptive gate (decide feature extraction mode)
        descriptors = compute_descriptors(pil_img, mask=mask)
        decision = gbdt_gate(descriptors, gbdt_model_path)

        # Step 4: Feature extraction with DINOv2
        feats = extract_spatial_features(np_img, strategy=decision)


        # Step 4: Core anomaly detection
        anomaly_map, score = compute_anomaly_scores(feats)

        # Step 5: Refinement / postprocess
        refined_map = refine_anomaly_map(anomaly_map)
        boosted_map = boosting_refinement(refined_map)

        # Collect outputs for evaluation
        all_scores.append(score)
        # all_labels.append(label)  # uncomment if labels available
        # all_masks.append(mask)    # uncomment if masks available

    # Step 6: Evaluation (if labels available)
    if len(all_labels) > 0:
        metrics = evaluate_results(all_scores, all_labels, all_masks)
        print("[RESULTS]", metrics)
    else:
        print("[INFO] Pipeline finished. No labels provided, skipping evaluation.")

if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2"
    CATEGORY = "can"
    SPLIT = "validation"  # since your dataset has validation instead of test
    GBDT_MODEL = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\src\models\gbdt_gate.pkl"

    run_pipeline(DATA_ROOT, CATEGORY, SPLIT, GBDT_MODEL)
