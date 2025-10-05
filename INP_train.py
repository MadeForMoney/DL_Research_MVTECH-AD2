# corrected_inp_train.py
import os
import traceback
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------
# Settings (edit if needed)
# ---------------------------
DATA_ROOT = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2"
CATEGORY = "can"  # Hardcoded category 'can'

# try common layout variants for validation folder
VALIDATE_PATH_CANDIDATES = [
    os.path.join(DATA_ROOT, CATEGORY, CATEGORY, "validation", "good"),
    os.path.join(DATA_ROOT, CATEGORY, "validation", "good"),
    os.path.join(DATA_ROOT, CATEGORY, CATEGORY, "validate", "good"),
    os.path.join(DATA_ROOT, CATEGORY, "validate", "good"),
]

NUM_IMAGES = 10  # how many images to sample from that folder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINO feature extraction params
DINO_BASE_SIZE = 518
DINO_PATCH_SIZE = 14  # will be read from model if available

# INP training params
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-4

# CPR bank
MAX_BANK_SIZE = 100000   # total patch vectors to store (subsample if needed)

# Save paths (use script folder)
SCRIPT_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "Inp_train1.pth")
BANK_SAVE_PATH = os.path.join(SCRIPT_DIR, "models", "cpr_bank.npy")

# ---------------------------
# DINOv2 feature extraction
# ---------------------------
_feature_extractor = None

def load_dinov2_model():
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor
    print(f"Loading DINOv2 model (torch.hub) on device {DEVICE}", flush=True)
    try:
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(DEVICE)
        _feature_extractor.eval()
        patch_sz = getattr(_feature_extractor.patch_embed, "patch_size", None)
        if isinstance(patch_sz, (tuple, list)):
            patch_sz = patch_sz[0]
        print(f"Loaded DINOv2. patch_size={patch_sz}", flush=True)
    except Exception as e:
        print("Failed to load DINOv2 model (network or hub issue). Falling back to dummy features.", flush=True)
        print("Error:", e, flush=True)
        _feature_extractor = None
    return _feature_extractor

def preprocess_image(img_np: np.ndarray, size: tuple):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return t(img_np).unsqueeze(0).to(DEVICE)

def extract_spatial_features(img_np: np.ndarray, strategy="multi-scale"):
    """Return numpy array (grid_h, grid_w, C). If DINO not available, return random features."""
    model = load_dinov2_model()
    if model is None:
        # Dummy features: choose grid ~ DINO expected base/patch
        grid = DINO_BASE_SIZE // DINO_PATCH_SIZE
        feat_dim = 768 * (3 if strategy == "multi-scale" else 1)
        return np.random.randn(grid, grid, feat_dim).astype(np.float32)

    patch_size = model.patch_embed.patch_size[0]
    with torch.no_grad():
        if strategy == "single-scale":
            size = (DINO_BASE_SIZE, DINO_BASE_SIZE)
            img_t = preprocess_image(img_np, size)
            features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
        else:
            scales = [1.0, 0.75, 0.5]
            base = DINO_BASE_SIZE
            feats = []
            for s in scales:
                h = int(base * s) // patch_size * patch_size
                w = int(base * s) // patch_size * patch_size
                if h == 0 or w == 0:
                    continue
                img_t = preprocess_image(img_np, (h, w))
                f = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]  # (1,C,h_p,w_p)
                f = torch.nn.functional.interpolate(
                    f, size=(base // patch_size, base // patch_size), mode='bilinear', align_corners=False
                )
                feats.append(f)
            features = torch.cat(feats, dim=1)  # concat channel-wise
    # features shape (1, C, H_grid, W_grid)
    features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return features

# ---------------------------
# INP-Former (same as your simplified version)
# ---------------------------
class MockTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x, *args, **kwargs):
        return x + self.ffn(self.norm(x))

class INP_Former(nn.Module):
    def __init__(self, feature_dim, grid_h, grid_w, num_layers=1):
        super().__init__()
        self.bottleneck = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.aggregation = nn.ModuleList([MockTransformerBlock(feature_dim)])
        self.decoder = nn.ModuleList([MockTransformerBlock(feature_dim) for _ in range(num_layers)])
        self.prototype_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feature_dim = feature_dim

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).permute(0, 2, 1)  # (B, L, C)
        # Aggregation
        agg_proto = self.prototype_token.repeat(B, 1, 1)
        for blk in self.aggregation:
            # MockTransformerBlock ignores extra args, so this is safe
            agg_proto = blk(agg_proto, patch_tokens)
        # Bottleneck
        z = patch_tokens
        for blk in self.bottleneck:
            z = blk(z)
        # Decoder
        de_list = []
        for blk in self.decoder:
            d = blk(z, agg_proto)
            de_list.append(d)
        # fuse decoder outputs
        de_fused = torch.stack(de_list, dim=1).mean(dim=1)
        # reshape to (B, C, H, W)
        de_map = de_fused.permute(0, 2, 1).reshape(B, C, H, W)
        return None, de_map  # we only need reconstructed map for training/inference

# ---------------------------
# Dataset helper for feature maps
# ---------------------------
class FeatureMapDataset(torch.utils.data.Dataset):
    def __init__(self, features_list):  # list of numpy arrays (H, W, C)
        self.features = [torch.from_numpy(f).permute(2,0,1).float() for f in features_list]  # (C, H, W)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

# ---------------------------
# Training loop for INP
# ---------------------------
def train_inp_with_features(feature_maps_np, grid_h, grid_w, feature_dim,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE):
    """
    feature_maps_np: list of numpy arrays (H, W, C) all same H, W, C
    returns trained INP model
    """
    dataset = FeatureMapDataset(feature_maps_np)
    print("DEBUG: Training dataset size =", len(dataset), flush=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = INP_Former(feature_dim, grid_h, grid_w, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print("Starting INP training...", flush=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        epoch_loss = running_loss / len(dataset) if len(dataset) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}", flush=True)
    print("INP training finished.", flush=True)
    # save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved INP model to", MODEL_PATH, flush=True)
    return model

# ---------------------------
# CPR bank helpers (patch-level)
# ---------------------------
def build_cpr_bank(feature_maps_np, max_bank_size=MAX_BANK_SIZE, seed=42):
    """
    feature_maps_np: list of numpy arrays (H, W, C)
    Returns: bank array shape (N_bank, C) and saved to disk
    """
    rng = np.random.RandomState(seed)
    all_patches = []
    for f in feature_maps_np:
        H, W, C = f.shape
        patches = f.reshape(-1, C)  # (H*W, C)
        all_patches.append(patches)
    all_patches = np.vstack(all_patches)  # (total_patches, C)
    total = all_patches.shape[0]
    if total <= max_bank_size:
        bank = all_patches.astype(np.float32)
    else:
        idx = rng.choice(total, size=max_bank_size, replace=False)
        bank = all_patches[idx].astype(np.float32)
    # Normalize for cosine retrieval
    norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8
    bank_normed = bank / norms
    os.makedirs(os.path.dirname(BANK_SAVE_PATH), exist_ok=True)
    np.save(BANK_SAVE_PATH, bank)          # store raw vectors
    np.save(BANK_SAVE_PATH.replace(".npy", "_norm.npy"), bank_normed)
    print(f"Saved CPR bank: {BANK_SAVE_PATH}  (size={bank.shape[0]} x {bank.shape[1]})", flush=True)
    return bank, bank_normed

def cpr_reconstruct_from_bank(feature_map_np, bank_normed, bank_raw):
    """
    Reconstruct a feature map by nearest-neighbor lookup (cosine similarity) into bank.
    """
    device = DEVICE
    # flatten queries
    L, C = feature_map_np.reshape(-1, feature_map_np.shape[2]).shape
    q = torch.from_numpy(feature_map_np.reshape(-1, feature_map_np.shape[2])).to(device).float()
    q_norm = F.normalize(q, dim=1)
    bank_t = torch.from_numpy(bank_normed).to(device).float().t()  # (C, N_bank)
    sims = torch.matmul(q_norm, bank_t)  # (L, N_bank)
    top1 = torch.argmax(sims, dim=1)
    bank_raw_t = torch.from_numpy(bank_raw).to(device).float()
    recon = bank_raw_t[top1].cpu().numpy()
    H, W, C = feature_map_np.shape
    recon_map = recon.reshape(H, W, C)
    return recon_map

# ---------------------------
# Utility: gather image paths
# ---------------------------
def gather_training_images(folder_path, num_images=NUM_IMAGES):
    """
    Gathers up to `num_images` image paths from `folder_path`.
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    if not os.path.exists(folder_path):
        raise RuntimeError(f"No folder found at {folder_path}")
    all_images = [os.path.join(folder_path, f)
                  for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in image_extensions]
    if len(all_images) == 0:
        raise RuntimeError(f"No image files found in {folder_path}")
    selected = sorted(all_images)[:min(num_images, len(all_images))]
    print(f"Selected {len(selected)} images for training from {folder_path}", flush=True)
    return selected

# ---------------------------
# Main orchestration
# ---------------------------
if __name__ == "__main__":
    try:
        # 0) Find a valid validation path from candidates
        VALIDATE_PATH = None
        for p in VALIDATE_PATH_CANDIDATES:
            if os.path.exists(p) and os.path.isdir(p):
                VALIDATE_PATH = p
                break
        if VALIDATE_PATH is None:
            # fallback: try DATA_ROOT/CATEGORY/*validation* directories
            matches = []
            for root, dirs, files in os.walk(os.path.join(DATA_ROOT, CATEGORY)):
                for d in dirs:
                    if "valid" in d.lower():
                        matches.append(os.path.join(root, d))
            if matches:
                VALIDATE_PATH = matches[0]
        if VALIDATE_PATH is None:
            # final fallback: use candidate[0] (so error message is clearer)
            VALIDATE_PATH = VALIDATE_PATH_CANDIDATES[0]

        print("Using validation folder:", VALIDATE_PATH, flush=True)

        # 1) Gather images from the specified folder
        img_paths = gather_training_images(VALIDATE_PATH, NUM_IMAGES)
        print("DEBUG: img_paths collected =", img_paths, flush=True)

        # 2) Extract DINO features for all training images
        print("Extracting DINOv2 features for training images (this may take time).", flush=True)
        feats = []
        for p in tqdm(img_paths, desc="DINO extract"):
            img_np = np.array(Image.open(p).convert("RGB"))
            feat = extract_spatial_features(img_np, strategy="multi-scale")  # (H_grid, W_grid, C)
            feats.append(feat.astype(np.float32))

        # Use shapes from first feature
        grid_h, grid_w, feature_dim = feats[0].shape
        print("Feature grid:", grid_h, grid_w, "feature_dim:", feature_dim, flush=True)

        # 3) Train INP on these feature maps
        inp_model = train_inp_with_features(feats, grid_h, grid_w, feature_dim,
                                            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE)

        # 4) Build CPR bank from original DINO features (not INP reconstructions)
        print("Building CPR bank (subsampling patches)...", flush=True)
        bank_raw, bank_normed = build_cpr_bank(feats, max_bank_size=MAX_BANK_SIZE)

        # 5) Save a small test: run one image through INP and CPR retrieval to create anomaly map
        test_idx = 0
        test_feat = feats[test_idx]  # numpy (H, W, C)
        inp_model.eval()
        with torch.no_grad():
            t = torch.from_numpy(test_feat).permute(2,0,1).unsqueeze(0).float().to(DEVICE)  # (1, C, H, W)
            _, recon_t = inp_model(t)
            recon_np = recon_t.squeeze(0).permute(1,2,0).cpu().numpy()  # (H, W, C)

        bank_norm_path = BANK_SAVE_PATH.replace(".npy", "_norm.npy")
        bank_norm_loaded = np.load(bank_norm_path)
        bank_raw_loaded = np.load(BANK_SAVE_PATH)
        recon_from_bank = cpr_reconstruct_from_bank(test_feat, bank_norm_loaded, bank_raw_loaded)  # (H, W, C)

        amap = np.mean((recon_np - recon_from_bank)**2, axis=2)  # (H, W)
        # upsample to original image size
        orig_img = np.array(Image.open(img_paths[test_idx]).convert("RGB"))
        H_orig, W_orig = orig_img.shape[:2]
        amap_up = F.interpolate(torch.from_numpy(amap).unsqueeze(0).unsqueeze(0),
                                size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze().numpy()
        amap_norm = (amap_up - amap_up.min()) / (np.ptp(amap_up) + 1e-8)
        amap_img = (255 * amap_norm).astype(np.uint8)

        out_path = os.path.join(SCRIPT_DIR, "models", "anomaly_test_inp_cpr.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import cv2
        cv2.imwrite(out_path, amap_img)
        print("Saved test anomaly map to:", out_path, flush=True)

        print("All done. Trained INP saved to:", MODEL_PATH, "CPR bank saved to:", BANK_SAVE_PATH, flush=True)

    except Exception as e:
        print("ERROR during execution:", str(e), flush=True)
        traceback.print_exc()
        raise
