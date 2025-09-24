import torch
import torch.nn as nn
import numpy as np
import cv2

class CNNRefiner(nn.Module):
    """
    Refines a binary mask using a simple U-Net-like CNN architecture.
    This helps in correcting boundaries and filling small artifacts.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder part
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        
        # Decoder part
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(16, out_channels, 3, padding=1), nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Decoder
        d1 = self.dec1(e2)
        d2 = self.dec2(d1)
        
        return d2

def cnn_based_refinement(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Wrapper function to apply the CNN-based refiner.
    (In a real scenario, the original_image might also be an input to the model).
    """
    model = CNNRefiner()
    model.eval()

    # Prepare the mask tensor
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        refined_tensor = model(mask_tensor)

    # Convert back to a binary numpy mask
    refined_mask = (refined_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return refined_mask

if __name__ == '__main__':
    # Create a dummy binary mask from post-processing
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (30, 30), (100, 100), 255, -1)
    cv2.circle(dummy_mask, (80, 80), 10, 0, -1) # Add a hole

    # Dummy original image (not used by this simplified model but needed for the function signature)
    dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)

    refined = cnn_based_refinement(dummy_mask, dummy_image)
    
    print("CNN-Based Refinement Demo")
    print(f"Input mask shape: {dummy_mask.shape}")
    print(f"Refined mask shape: {refined.shape}")
    # cv2.imwrite("cnn_refined_mask.png", refined)
