import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# INP Model Definition
# ----------------------------
class MockTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    def forward(self, x, *args, **kwargs):
        return x + self.ffn(self.norm(x))

class INP_Former(nn.Module):
    def __init__(self, feature_dim, grid_h, grid_w, num_layers=2):
        super().__init__()
        self.bottleneck = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.aggregation = nn.ModuleList([MockTransformerBlock(feature_dim)])
        self.decoder = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.prototype_token = nn.Parameter(torch.randn(1,1,feature_dim))
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feature_dim = feature_dim

    def forward(self, x):
        B,C,H,W = x.shape
        patch_tokens = x.flatten(2).permute(0,2,1)
        agg_proto = self.prototype_token.repeat(B,1,1)
        for blk in self.aggregation:
            agg_proto = blk(agg_proto, patch_tokens)
        z = patch_tokens
        for blk in self.bottleneck:
            z = blk(z)
        de_list=[]
        for blk in self.decoder:
            d = blk(z, agg_proto)
            de_list.append(d)
        de_fused = torch.stack(de_list, dim=1).mean(dim=1)
        de_map = de_fused.permute(0,2,1).reshape(B,C,H,W)
        return None, de_map

# ----------------------------
# Load trained INP + CPR
# ----------------------------
def load_trained_models(model_dir, feature_dim, grid_h, grid_w):
    inp_path = os.path.join(model_dir, "Inp_train1.pth")
    bank_raw_path = os.path.join(model_dir, "cpr_bank.npy")
    bank_normed_path = os.path.join(model_dir, "cpr_bank_norm.npy")

    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"INP_Former model not found: {inp_path}")
    if not os.path.exists(bank_raw_path) or not os.path.exists(bank_normed_path):
        raise FileNotFoundError(f"CPR bank or normalized bank not found in {model_dir}")

    print(f"✅ Loading INP_Former from {inp_path}")
    inp_model = INP_Former(feature_dim, grid_h, grid_w).to(device)
    state = torch.load(inp_path, map_location=device)
    inp_model.load_state_dict(state)
    inp_model.eval()

    print(f"✅ Loading CPR banks")
    bank_raw = np.load(bank_raw_path)
    bank_normed = np.load(bank_normed_path)

    return inp_model, bank_raw, bank_normed

# ----------------------------
# CPR Reconstruction
# ----------------------------
def cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    L,C = feature_map_np.reshape(-1, feature_map_np.shape[2]).shape
    q = torch.from_numpy(feature_map_np.reshape(-1,C)).to(device).float()
    q_norm = F.normalize(q, dim=1)
    bank_t = torch.from_numpy(bank_normed).to(device).float().t()
    sims = torch.matmul(q_norm, bank_t)
    top1 = torch.argmax(sims, dim=1)
    bank_raw_t = torch.from_numpy(bank_raw).to(device).float()
    recon = bank_raw_t[top1].cpu().numpy()
    H,W,C = feature_map_np.shape
    return recon.reshape(H,W,C)

# ----------------------------
# Compute anomaly map
# ----------------------------
def compute_anomaly_map(feature_map_np, inp_model, bank_normed, bank_raw, output_shape):
    # INP reconstruction
    t = torch.from_numpy(feature_map_np).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        _, recon = inp_model(t)
    recon_np = recon.squeeze(0).permute(1,2,0).cpu().numpy()

    # CPR reconstruction
    recon_bank = cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw)

    # Per-pixel anomaly (MSE)
    amap = np.mean((recon_np - recon_bank)**2, axis=2)

    # Resize to original image
    H_orig, W_orig = output_shape
    amap_up = F.interpolate(torch.from_numpy(amap).unsqueeze(0).unsqueeze(0),
                            size=(H_orig, W_orig),
                            mode='bilinear',
                            align_corners=False).squeeze().numpy()

    # Normalize to 0-255
    prob_map = (amap_up - amap_up.min()) / (amap_up.max() - amap_up.min() + 1e-8)
    prob_map_uint8 = (prob_map * 255).astype(np.uint8)

    # Compute a single anomaly score (mean normalized anomaly)
    anomaly_score = prob_map.mean() / 255.0  # 0-1 scale

    return prob_map_uint8, anomaly_score

# ----------------------------
# Overlay heatmap
# ----------------------------
def overlay_heatmap(image_np, anomaly_map, alpha=0.6):
    heatmap_color = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, image_np, 1-alpha, 0)
    return overlay

# ----------------------------
# Full inference
# ----------------------------
def run_inference(img_np, model_dir, feature_map_np):
    H_feat, W_feat, C_feat = feature_map_np.shape
    inp_model, bank_raw, bank_normed = load_trained_models(
        model_dir, feature_dim=C_feat, grid_h=H_feat, grid_w=W_feat
    )
    anomaly_map, anomaly_score = compute_anomaly_map(feature_map_np, inp_model, bank_normed, bank_raw, img_np.shape[:2])
    overlay = overlay_heatmap(img_np, anomaly_map)
    return anomaly_map, overlay, anomaly_score


# ----------------------------
# Example usage
# ----------------------------
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
    anomaly_map, overlay, score = run_inference(img, MODEL_DIR, feature_map_np)
    print(anomaly_map)
    print("Anomaly score (0-1):", score)


    # Save outputs
    # OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "anomaly_map.png"), anomaly_map)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("✅ Inference completed!")
