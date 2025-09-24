import torch
import torch.nn as nn
import numpy as np

class TransformerRefiner(nn.Module):
    """
    Refines a mask by treating it as a sequence of patches and using a Transformer.
    This allows it to capture global context for better boundary localization.
    """
    def __init__(self, patch_size=16, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. Embed patches
        x = self.patch_embed(x) # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        
        # 2. Process with Transformer
        refined_tokens = self.transformer_encoder(x)
        
        # 3. Reconstruct the image
        patch_outputs = self.output_head(refined_tokens) # (B, num_patches, p*p)
        patch_outputs = torch.sigmoid(patch_outputs)
        
        # Fold patches back into an image (B, num_patches, p*p) -> (B, 1, H, W)
        x = patch_outputs.reshape(B, H // self.patch_size, W // self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, H, W).unsqueeze(1)
        
        return x

def transformer_based_refinement(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    model = TransformerRefiner(patch_size=16)
    model.eval()
    
    h, w = mask.shape
    # Ensure divisible by patch size
    target_h, target_w = (h // 16) * 16, (w // 16) * 16
    mask_resized = cv2.resize(mask, (target_w, target_h))
    
    mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        refined_tensor = model(mask_tensor)
        
    refined_mask_resized = (refined_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    # Resize back to original dimensions
    refined_mask = cv2.resize(refined_mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)
    return refined_mask

if __name__ == '__main__':
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (30, 30), (100, 100), 255, -1)
    dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    refined = transformer_based_refinement(dummy_mask, dummy_image)
    
    print("Transformer-Based Refinement Demo")
    print(f"Refined mask shape: {refined.shape}")
    # cv2.imwrite("transformer_refined_mask.png", refined)
