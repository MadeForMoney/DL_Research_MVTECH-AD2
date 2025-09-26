import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import urllib

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global holder for model
_feature_extractor = None

def load_dinov2_model():
    """Loads DINOv2 (ViT-B/14) model via torch.hub if not already loaded."""
    global _feature_extractor
    if _feature_extractor is not None:
        return _feature_extractor

    print(f"Loading DINOv2 model onto {device}")
    try:
        # Load from torch.hub
        # Use one of the official entrypoints, e.g. 'dinov2_vitb14'
        _feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        _feature_extractor.eval()
        print("DINOv2 ViT-B/14 loaded successfully.")
    except (urllib.error.URLError, ConnectionError) as e:
        print("Failed to load DINOv2 model:", e)
        _feature_extractor = None

    return _feature_extractor

def preprocess_image(img_np: np.ndarray, size: tuple) -> torch.Tensor:
    """
    Preprocess a NumPy image (H×W×C, uint8 or float) for DINOv2.
    - Resize
    - ToTensor
    - Normalize (ImageNet mean/std)
    Returns a tensor on `device`, shape (1, C, H, W).
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img_np).unsqueeze(0).to(device)

def extract_spatial_features(img_np: np.ndarray, strategy: str = 'single-scale') -> np.ndarray:
    """
    Given an RGB image as NumPy (H, W, C), extract patch-level DINOv2 features.
    strategy: 'single-scale' or 'multi-scale'.
    Returns: spatial features as NumPy array of shape (grid_h, grid_w, feature_dim)
    """
    model = load_dinov2_model()
    if model is None:
        print("Model is not loaded. Returning zeros.")
        return np.zeros((1,1,768))

    patch_size = model.patch_embed.patch_size[0]  # e.g. 14

    with torch.no_grad():
        if strategy == 'single-scale':
            # choose a size divisible by patch_size
            size = (518, 518)
            img_t = preprocess_image(img_np, size)
            # get the last layer features, reshaped to spatial grid
            features = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
            # features: (1, feature_dim, grid_h, grid_w)
        else:
            # multi-scale
            scales = [1.0, 0.75, 0.5]
            base = 518
            feats = []
            for s in scales:
                h = int(base * s) // patch_size * patch_size
                w = int(base * s) // patch_size * patch_size
                if h == 0 or w == 0:
                    continue
                img_t = preprocess_image(img_np, (h, w))
                f = model.get_intermediate_layers(img_t, n=1, reshape=True)[0]
                # upsample spatially to base grid size
                f = torch.nn.functional.interpolate(
                    f,
                    size=(base // patch_size, base // patch_size),
                    mode='bilinear',
                    align_corners=False
                )
                feats.append(f)
            features = torch.cat(feats, dim=1)

    # Convert to NumPy spatial (grid_h, grid_w, channels)
    features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return features


if __name__ == "__main__":
    # Test with a random image
    dummy_path = '/content/003_regular.png'
    # Load the image as a NumPy array
    dummy_img_np = np.array(Image.open(dummy_path).convert('RGB'))


    print("Single-scale extraction...")
    feat1 = extract_spatial_features(dummy_img_np, strategy='single-scale')
    print("Output shape:", feat1.shape)

    print("Multi-scale extraction...")
    feat2 = extract_spatial_features(dummy_img_np, strategy='multi-scale')
    print("Output shape:", feat2.shape)
