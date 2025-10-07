import numpy as np
import cv2
from anomaly_core import run_inference

def postprocess_anomaly_map(raw_map: np.ndarray, threshold_pct: float = 0.1,
                            close_kernel_size: int = 3, open_kernel_size: int = 3):
    # Convert to float
    norm_map = raw_map.astype(np.float32)

    # Optional: amplify small differences
    norm_map *= 10.0

    # Normalize to 0-1
    norm_map = (norm_map - norm_map.min()) / (norm_map.max() - norm_map.min() + 1e-8)

    # Thresholding
    threshold_value = np.percentile(norm_map, 100 * (1 - threshold_pct))
    binary_mask = (norm_map >= threshold_value).astype(np.uint8) * 255

    # Morphological Closing
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

    # Morphological Opening
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)

    # Visualization map (0-255)
    vis_map = (norm_map * 255).astype(np.uint8)

    return opened_mask, vis_map




if __name__=="__main__":
    import os
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
    TEST_IMAGE_PATH = os.path.join(SCRIPT_DIR, "mvtec_ad_2", "can", "can", "test_public", "bad", "000_regular.png")

    # Load image
    img = cv2.imread(TEST_IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assume you already extracted features using DINOv2
    from feature_extract import extract_spatial_features
    feature_map_np = extract_spatial_features(img, strategy="multi-scale")

    # Run inference
    anomaly_map, overlay = run_inference(img, MODEL_DIR, feature_map_np)
    open_mask,normalized_map=postprocess_anomaly_map(anomaly_map)
    print(normalized_map)

    # OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "PostPorce.png"), normalized_map)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "Postover.png"), cv2.cvtColor(open_mask, cv2.COLOR_RGB2BGR))
