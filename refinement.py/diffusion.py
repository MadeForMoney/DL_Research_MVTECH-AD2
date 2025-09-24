import torch
import torch.nn as nn
import numpy as np

class DiffusionDenoisingRefiner(nn.Module):
    """
    A simple U-Net model that acts as a denoiser, which is the core of a
    diffusion model's reverse process. It learns to reconstruct a clean mask.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1 = self.dec1(self.upconv1(b))
        d2 = self.dec2(self.upconv2(d1))
        return d2

def diffusion_based_refinement(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    model = DiffusionDenoisingRefiner()
    model.eval()

    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        # A single step of denoising for demonstration
        refined_tensor = model(mask_tensor)

    refined_mask = (refined_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return refined_mask

if __name__ == '__main__':
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (30, 30), (100, 100), 255, -1)
    # Add significant noise to be "denoised"
    noise = np.random.randint(0, 2, (128, 128), dtype=np.uint8) * 255
    dummy_mask[noise > 0] = 0
    
    dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    refined = diffusion_based_refinement(dummy_mask, dummy_image)
    
    print("Diffusion-Based Refinement Demo")
    print(f"Refined mask shape: {refined.shape}")
    # cv2.imwrite("diffusion_refined_mask.png", refined)
