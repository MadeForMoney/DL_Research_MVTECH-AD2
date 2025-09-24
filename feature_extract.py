import torch
import cv2
import numpy as np
from torchvision import transforms
import urllib

# --- Model and Preprocessing Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None

def load_feature_extractor():
    """Loads the DINOv2 model if it hasn't been loaded yet."""
    global feature_extractor
    if feature_extractor is not None:
        return
        
    print(f"Using device: {device}")
    print("Loading DINOv2 feature extractor model...")
    try:
        # Load the DINOv2 model from PyTorch Hub
        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        feature_extractor.eval()
        print("DINOv2 ViT-Base model loaded successfully.")
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"Could not load DINOv2 model due to a network issue: {e}")
        feature_extractor = None

def _preprocess_image(image: np.ndarray, size: tuple) -> torch.Tensor:
    """Applies the standard DINOv2 preprocessing to an image."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

# --- Main Feature Extraction Function ---

def extract_spatial_features(image: np.ndarray, strategy: str = 'single-scale') -> np.ndarray:
    """
    Extracts spatial patch tokens from an image using DINOv2.

    The output retains spatial information necessary for generating anomaly maps.

    Args:
        image (np.ndarray): The input image in BGR format.
        strategy (str): Can be 'single-scale' or 'multi-scale', determined by the GBDT gate.

    Returns:
        np.ndarray: A 3D numpy array of shape (grid_h, grid_w, feature_dim), 
                    representing the grid of patch features.
    """
    load_feature_extractor()
    if feature_extractor is None:
        print("Warning: Model not loaded. Returning a zero array.")
        return np.zeros((1, 1, 768))

    patch_dim = feature_extractor.patch_embed.patch_size[0] # Usually 14 for ViT-B/14

    with torch.no_grad():
        if strategy == 'single-scale':
            # Use a standard, high-resolution size for single-scale
            h, w = 518, 518 # A size divisible by the patch size (14)
            image_tensor = _preprocess_image(image, size=(h, w))
            
            # Use get_intermediate_layers to get the grid of patch tokens
            # n=1 gets the last layer; reshape=True returns it as a spatial grid
            features_tensor = feature_extractor.get_intermediate_layers(image_tensor, n=1, reshape=True)[0]
            # Output shape: (1, 768, H/14, W/14)
            
        else: # 'multi-scale' strategy
            scales = [1.0, 0.75, 0.5]
            all_features = []
            base_size = 518
            
            for scale in scales:
                # Calculate target size that is a multiple of the patch dimension
                h = int(base_size * scale) // patch_dim * patch_dim
                w = int(base_size * scale) // patch_dim * patch_dim
                if h == 0 or w == 0: continue
                
                image_tensor = _preprocess_image(image, size=(h, w))
                scaled_features = feature_extractor.get_intermediate_layers(image_tensor, n=1, reshape=True)[0]
                
                # Upsample the feature maps to the base size to allow concatenation
                upsampled_features = torch.nn.functional.interpolate(
                    scaled_features,
                    size=(base_size // patch_dim, base_size // patch_dim),
                    mode='bilinear',
                    align_corners=False
                )
                all_features.append(upsampled_features)
            
            # Concatenate features along the channel dimension
            features_tensor = torch.cat(all_features, dim=1)
            # Output shape: (1, 768*3, H/14, W/14)

    # Convert tensor to numpy array and move channel dim to the end: (H, W, C)
    spatial_features = features_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return spatial_features

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy BGR image
    dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    print(f"Created a dummy image of shape: {dummy_image.shape}\n")

    # --- Test Single-Scale Spatial Feature Extraction ---
    print("--- Extracting single-scale spatial features ---")
    single_scale_spatial_features = extract_spatial_features(dummy_image, strategy='single-scale')
    print(f"Shape of output: {single_scale_spatial_features.shape}")
    print("This 3D array represents a grid of feature vectors, ready for anomaly map generation.")
    # Expected shape for a 518x518 input: (37, 37, 768)
    
    # --- Test Multi-Scale Spatial Feature Extraction ---
    print("\n--- Extracting multi-scale spatial features ---")
    multi_scale_spatial_features = extract_spatial_features(dummy_image, strategy='multi-scale')
    print(f"Shape of output: {multi_scale_spatial_features.shape}")
    print("The feature dimension is now 3x larger, as it concatenates features from all scales.")
    # Expected shape: (37, 37, 2304)
