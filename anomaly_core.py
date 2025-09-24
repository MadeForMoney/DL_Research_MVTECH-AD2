# # import torch
# # import torch.nn as nn
# # import numpy as np

# # # --- 1. Define Placeholder Model Components ---

# # class INPFormer(nn.Module):
# #     """
# #     A placeholder for the 'INP-Former' model component.
    
# #     This likely would be a Transformer-based network for feature reconstruction
# #     or in-painting. Here, we simulate it with a simple feed-forward network.
# #     """
# #     def __init__(self, feature_dim=2304):
# #         super().__init__()
# #         self.model = nn.Sequential(
# #             nn.Linear(feature_dim, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, feature_dim) # Outputs reconstructed features
# #         )
    
# #     def forward(self, features):
# #         reconstructed_features = self.model(features)
# #         # The anomaly contribution could be the reconstruction error
# #         reconstruction_error = torch.mean((features - reconstructed_features)**2, dim=-1)
# #         return reconstructed_features, reconstruction_error

# # class CPR(nn.Module):
# #     """
# #     A placeholder for the 'CPR' (Contextual Patch Refinement, etc.) model.
    
# #     This stage likely refines the initial anomaly score using the original and
# #     reconstructed features.
# #     """
# #     def __init__(self, feature_dim=2304):
# #         super().__init__()
# #         # Takes original + reconstructed features concatenated
# #         self.model = nn.Sequential(
# #             nn.Linear(feature_dim * 2, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 1) # Outputs a final refinement score
# #         )
        
# #     def forward(self, original_features, reconstructed_features):
# #         combined_features = torch.cat((original_features, reconstructed_features), dim=-1)
# #         refinement_score = self.model(combined_features)
# #         return refinement_score.squeeze(-1)

# # # --- 2. Define the Hybrid Model ---

# # class HybridAnomalyModel(nn.Module):
# #     """
# #     Combines the INP-Former and CPR into a single hybrid model.
# #     """
# #     def __init__(self, feature_dim=2304):
# #         super().__init__()
# #         self.inp_former = INPFormer(feature_dim)
# #         self.cpr = CPR(feature_dim)
        
# #     def forward(self, features):
# #         # Stage 1: Get reconstructed features and initial error from INP-Former
# #         reconstructed_features, initial_error = self.inp_former(features)
        
# #         # Stage 2: Refine the score using CPR
# #         refinement_score = self.cpr(features, reconstructed_features)
        
# #         # Combine the scores (e.g., by adding them)
# #         final_anomaly_score = initial_error + refinement_score
# #         return final_anomaly_score

# # # --- 3. Fill in the Main Function ---

# # def run_anomaly_detection(features: np.ndarray) -> float:
# #     """
# #     Runs the hybrid model inference (INP-Former + CPR) on a set of features.

# #     Args:
# #         features (np.ndarray): A 1D numpy array of features extracted from an image.

# #     Returns:
# #         float: The final anomaly score for the input features.
# #     """
# #     # Assuming 'features' is a 1D numpy array, e.g., from a multi-scale extractor
# #     feature_dim = features.shape[0]
    
# #     # --- Model Loading and Preparation ---
# #     # In a real scenario, you would load pre-trained weights here
# #     # model.load_state_dict(torch.load('path/to/your/weights.pth'))
# #     model = HybridAnomalyModel(feature_dim=feature_dim)
# #     model.eval() # Set the model to evaluation mode
    
# #     # --- Inference ---
# #     # Convert numpy features to a PyTorch tensor
# #     features_tensor = torch.from_numpy(features).float().unsqueeze(0) # Add batch dimension
    
# #     # Perform inference without calculating gradients
# #     with torch.no_grad():
# #         anomaly_score = model(features_tensor)
        
# #     # Return the score as a standard Python float
# #     return anomaly_score.item()


# # # --- 4. Example Usage ---
# # if __name__ == '__main__':
# #     # Let's assume our features come from a multi-scale ViT (e.g., 3 scales * 768 dims)
# #     feature_dimension = 2304
    
# #     # Create a dummy feature vector to simulate input
# #     print(f"Creating a dummy feature vector of size {feature_dimension}...")
# #     dummy_features = np.random.rand(feature_dimension).astype(np.float32)
    
# #     # Run the anomaly detection
# #     print("Running hybrid anomaly detection...")
# #     final_score = run_anomaly_detection(dummy_features)
    
# #     print(f"\nDetection complete.")
# #     print(f"Final Anomaly Score: {final_score:.4f}")
    
# #     # Example of a slightly different vector (potential anomaly)
# #     print("\nRunning on a slightly different feature vector...")
# #     anomalous_features = dummy_features.copy()
# #     anomalous_features[100:200] *= 5 # Exaggerate some feature values
# #     final_score_anomalous = run_anomaly_detection(anomalous_features)
    
# #     print(f"Anomalous Feature Score: {final_score_anomalous:.4f}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# from functools import partial
# from typing import List, Union, Dict
# from torchvision.models import resnet

# # #############################################################################
# # SECTION 1: Placeholder Modules Required by INP_Former
# # These are simplified stand-ins for the complex components INP_Former needs.
# # #############################################################################

# class MockTransformerBlock(nn.Module):
#     """A simple placeholder for a Transformer block."""
#     def __init__(self, dim=768):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    
#     def forward(self, x, *args, **kwargs):
#         x = x + self.ffn(self.norm(x))
#         return x

# class MockEncoder(nn.Module):
#     """A placeholder for the DINOv2 encoder."""
#     def __init__(self, img_size=320, patch_size=16, embed_dim=768, depth=12):
#         super().__init__()
#         self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.blocks = nn.ModuleList([MockTransformerBlock(embed_dim) for _ in range(depth)])
#         self.num_register_tokens = 0
        
#     def prepare_tokens(self, x):
#         x = self.patch_embed(x).flatten(2).transpose(1, 2)
#         return x

# # #############################################################################
# # SECTION 2: Your Exact INP_Former and CPR Code
# # This is the code you provided, integrated directly.
# # #############################################################################

# # --- Your INP_Former Code ---
# class INP_Former(nn.Module):
#     def __init__(
#             self,
#             encoder,
#             bottleneck,
#             aggregation,
#             decoder,
#             target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
#             fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
#             fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
#             remove_class_token=False,
#             encoder_require_grad_layer=[],
#             prototype_token=None,
#     ) -> None:
#         super(INP_Former, self).__init__()
#         self.encoder = encoder
#         self.bottleneck = bottleneck
#         self.aggregation = aggregation
#         self.decoder = decoder
#         self.target_layers = target_layers
#         self.fuse_layer_encoder = fuse_layer_encoder
#         self.fuse_layer_decoder = fuse_layer_decoder
#         self.remove_class_token = remove_class_token
#         self.encoder_require_grad_layer = encoder_require_grad_layer
#         self.prototype_token = prototype_token[0]
#         if not hasattr(self.encoder, 'num_register_tokens'):
#             self.encoder.num_register_tokens = 0
            
#     def gather_loss(self, query, keys):
#         self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
#         self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
#         gather_loss = self.distance.mean()
#         return gather_loss

#     def forward(self, x, use_gather_loss=True, return_patch_tokens=False):
#         x = self.encoder.prepare_tokens(x)
#         B, L, C = x.shape
#         en_list = []
#         for i, blk in enumerate(self.encoder.blocks):
#             if i <= self.target_layers[-1]:
#                 x = blk(x)
#             else:
#                 continue
#             if i in self.target_layers:
#                 en_list.append(x)
#         side = int(math.sqrt(L)) # Adjusted to use L from shape
#         if self.remove_class_token:
#             en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
#         patch_tokens = self.fuse_feature(en_list)
#         agg_prototype = self.prototype_token
#         for i, blk in enumerate(self.aggregation):
#             agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), patch_tokens)
#         if use_gather_loss:
#             g_loss = self.gather_loss(patch_tokens, agg_prototype)
#         else:
#             g_loss = torch.tensor(0.0, device=patch_tokens.device)
#         x = patch_tokens
#         for i, blk in enumerate(self.bottleneck):
#             x = blk(x)
#         de_list = []
#         for i, blk in enumerate(self.decoder):
#             x = blk(x, agg_prototype)
#             de_list.append(x)
#         de_list = de_list[::-1]
#         en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
#         de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
#         if not self.remove_class_token:
#             if self.encoder.num_register_tokens > 0:
#                  en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
#                  de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]
#         en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
#         de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]
#         if return_patch_tokens:
#             return en, de, g_loss, patch_tokens, agg_prototype
#         return en, de, g_loss, agg_prototype

#     def fuse_feature(self, feat_list):
#         return torch.stack(feat_list, dim=1).mean(dim=1)

# # --- Your CPR Code ---
# class LastLayerToExtractReachedException(Exception): pass

# class ForwardHook:
#     def __init__(self, stop: bool = False):
#         self._feature = None
#         self.stop = stop
#     def __call__(self, module, input, output):
#         self._feature = output
#         if self.stop: raise LastLayerToExtractReachedException()
#     @property
#     def feature(self):
#         try: return self._feature
#         finally: self._feature = None

# class FeatureExtractor(nn.Module):
#     def __init__(self, backbone: nn.Module, layers: Union[List[str], Dict[str, str]]) -> None:
#         super().__init__()
#         self.shapes: Dict[str, torch.Size] = None
#         self.backbone = backbone
#         self.forward_hooks: List[ForwardHook] = []
#         self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
#         self.layers = isinstance(layers, dict) and layers or dict(zip(layers, layers))
#         for idx, layer in enumerate(self.layers):
#             forward_hook = ForwardHook(idx == len(self.layers) - 1)
#             network_layer = backbone
#             while "." in layer:
#                 extract_block, layer = layer.split(".", 1)
#                 network_layer = network_layer.__dict__["_modules"][extract_block]
#             network_layer = network_layer.__dict__["_modules"][layer]
#             self.hook_handles.append(network_layer.register_forward_hook(forward_hook))
#             self.forward_hooks.append(forward_hook)
            
#     def forward(self, *x):
#         try: self.backbone(*x)
#         except LastLayerToExtractReachedException: pass
#         finally: return dict(zip(self.layers.values(), map(lambda hook: hook.feature, self.forward_hooks)))

# class BaseModel(nn.Module):
#     def __init__(self, layers: List[str] = None, backbone_name: str = None, input_size: int = 320):
#         super().__init__()
#         self.layers = layers
#         self.backbone_name = backbone_name
#         self.feature_extractor = FeatureExtractor(self.load_backbone(), self.layers)
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, input_size, input_size)
#             self.shapes = [f.shape for f in self(dummy_input)]

#     def forward(self, x) -> List[torch.Tensor]:
#         return list(self.feature_extractor(x).values())
    
#     def load_backbone(self): raise NotImplementedError()

# class ResNet(BaseModel):
#     def __init__(self, layers: List[str] = ['layer1', 'layer2'], backbone_name: str = 'resnet18', **kwargs):
#         super().__init__(layers, backbone_name, **kwargs)
    
#     def load_backbone(self) -> resnet.ResNet:
#         return getattr(resnet, self.backbone_name)(pretrained=True)

# class Inception(nn.Module):
#     def __init__(self, in_channels: int = 192, out_channels: int = 256):
#         super().__init__()
#         self.branch0 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
#         self.branch1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
#         self.branch2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
#         self.branch3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
#     def forward(self, x): return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1)

# class LocalRetrievalBranch(nn.Module):
#     def __init__(self, in_channels_list: List[int], out_channels_list: List[int]) -> None:
#         super().__init__()
#         self.conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels, out_channels in zip(in_channels_list, out_channels_list)])
#     def forward(self, xs: List[torch.Tensor]):
#         return [layer(x) for x, layer in zip(xs, self.conv)]

# class CPR(nn.Module):
#     def __init__(self, backbone: BaseModel, lrb: LocalRetrievalBranch) -> None:
#         super().__init__()
#         self.lrb = lrb
#         self.backbone = backbone
#     def forward(self, x):
#         ori_features = self.backbone(x)
#         return self.lrb(ori_features), ori_features

# def create_cpr_model(model_name: str = 'ResNet', layers: List[str] = ['layer1', 'layer2'], input_size: int = 320, output_dim: int = 384) -> CPR:
#     model_infos = {'ResNet': {'layers': layers, 'cls': ResNet}}
#     backbone: BaseModel = model_infos[model_name]['cls'](layers, input_size=input_size).eval()
#     lrb = LocalRetrievalBranch([shape[1] for shape in backbone.shapes], [output_dim] * len(layers))
#     return CPR(backbone, lrb)

# # #############################################################################
# # SECTION 3: The Hybrid Model and Main Inference Function
# # #############################################################################

# class HybridAnomalyModel(nn.Module):
#     def __init__(self, input_size=320, embed_dim=768):
#         super().__init__()
#         # 1. Initialize INP-Former with placeholder components
#         encoder = MockEncoder(img_size=input_size, embed_dim=embed_dim)
#         self.inp_former = INP_Former(
#             encoder=encoder,
#             bottleneck=nn.ModuleList([MockTransformerBlock(embed_dim)]),
#             aggregation=nn.ModuleList([MockTransformerBlock(embed_dim)]),
#             decoder=nn.ModuleList([MockTransformerBlock(embed_dim) for _ in range(2)]),
#             prototype_token=nn.Parameter(torch.randn(1, 1, embed_dim)),
#         )
        
#         # 2. Initialize CPR model using its creation function
#         self.cpr = create_cpr_model(input_size=input_size, output_dim=embed_dim)

#     def forward(self, x):
#         # --- Run both models in parallel ---
#         # Get encoder feature maps from INP-Former
#         inp_maps, _, _, _ = self.inp_former(x, use_gather_loss=False)
#         inp_map = inp_maps[0] # Take the first fused feature map
        
#         # Get original feature maps from CPR backbone
#         _, cpr_maps = self.cpr(x)
#         cpr_map = cpr_maps[0] # Take the first feature map from CPR
        
#         # --- Calculate Anomaly Score ---
#         # Resize maps to be the same size for comparison
#         cpr_map_resized = F.interpolate(cpr_map, size=inp_map.shape[-2:], mode='bilinear', align_corners=False)
        
#         # Calculate anomaly score as the mean squared error between the two feature maps
#         anomaly_map = (inp_map - cpr_map_resized) ** 2
#         score = torch.mean(anomaly_map)
        
#         return score

# def run_anomaly_detection(image: np.ndarray) -> float:
#     """
#     Runs the hybrid model inference (INP-Former + CPR) on an image.
#     Args:
#         image (np.ndarray): An input image in HxWxC format (e.g., 320x320x3).
#     Returns:
#         float: The final anomaly score.
#     """
#     # 1. Initialize the Hybrid Model
#     model = HybridAnomalyModel(input_size=image.shape[0])
#     model.eval()
    
#     # 2. Preprocess the image
#     # Convert numpy image (H, W, C) to PyTorch tensor (1, C, H, W)
#     image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
#     # Normalize (using standard ImageNet stats)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     image_tensor = normalize(image_tensor)
    
#     # 3. Run Inference
#     with torch.no_grad():
#         score = model(image_tensor)
        
#     return score.item()

# # #############################################################################
# # SECTION 4: Example Usage
# # #############################################################################
# if __name__ == '__main__':
#     from torchvision import transforms # Add transforms for preprocessing
    
#     # Define image size (must match model's expected input size)
#     INPUT_SIZE = 320 
    
#     # Create a dummy image
#     dummy_image = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    
#     print("--- Running Hybrid Anomaly Detection (INP-Former + CPR) ---")
    
#     # Run the main function
#     anomaly_score = run_anomaly_detection(dummy_image)
    
#     print(f"\nDetection complete.")
#     print(f"Final Anomaly Score: {anomaly_score:.6f}")


# File: anomaly_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List
import cv2

# #############################################################################
# SECTION 1: Your INP_Former and CPR Code, adapted for feature map inputs
# #############################################################################

class INP_Former(nn.Module):
    # This is your INP_Former. Its 'encoder' will now be a simple reshaping layer.
    def __init__(self, encoder, bottleneck, aggregation, decoder, prototype_token, target_layers, fuse_layer_encoder, fuse_layer_decoder):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.prototype_token = prototype_token
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
    
    def forward(self, x):
        # x is a feature map (B, C, H, W). The encoder reshapes it to patch tokens.
        patch_tokens = self.encoder(x) 
        B, L, C = patch_tokens.shape
        side = int(math.sqrt(L))

        # We only have one layer of features now, so en_list is simple.
        en_list = [patch_tokens] * len(self.target_layers)

        # Aggregation and Bottleneck
        agg_prototype = self.prototype_token
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat(B, 1, 1), patch_tokens)
        
        x = patch_tokens
        for blk in self.bottleneck:
            x = blk(x)
        
        # Decoder
        de_list = []
        for blk in self.decoder:
            x = blk(x, agg_prototype)
            de_list.append(x)
        
        # Fuse decoder features (encoder features are just the input)
        en_map = x.permute(0, 2, 1).reshape(B, C, side, side)
        de_fused = torch.stack(de_list, dim=1).mean(dim=1)
        de_map = de_fused.permute(0, 2, 1).reshape(B, C, side, side)
        
        return en_map, de_map

# The CPR's backbone is replaced with an Identity layer, as features are pre-computed.
class IdentityBackbone(nn.Module):
    def __init__(self, shapes):
        super().__init__(); self.shapes = shapes
    def forward(self, x):
        return [F.interpolate(x, size=s[-2:], mode='bilinear', align_corners=False) for s in self.shapes]

class LocalRetrievalBranch(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels_list: List[int]):
        super().__init__(); self.conv = nn.ModuleList([nn.Conv2d(c_in, c_out, 1) for c_in, c_out in zip(in_channels_list, out_channels_list)])
    def forward(self, xs: List[torch.Tensor]): return [layer(x) for x, layer in zip(xs, self.conv)]

class CPR(nn.Module):
    def __init__(self, backbone: nn.Module, lrb: LocalRetrievalBranch):
        super().__init__(); self.lrb = lrb; self.backbone = backbone
    def forward(self, x):
        ori_features = self.backbone(x)
        return self.lrb(ori_features), ori_features

# #############################################################################
# SECTION 2: The Hybrid Anomaly Model Wrapper
# #############################################################################
class MockTransformerBlock(nn.Module):
    def __init__(self, dim): super().__init__(); self.norm = nn.LayerNorm(dim); self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    def forward(self, x, *args, **kwargs): return x + self.ffn(self.norm(x))

class HybridAnomalyModel(nn.Module):
    """Wraps INP_Former and CPR to work on a pre-computed feature map."""
    def __init__(self, feature_dim, grid_size):
        super().__init__()
        # 1. INP-Former Setup
        # The new 'encoder' just reshapes the (B, C, H, W) map to (B, L, C) tokens.
        inp_encoder = nn.Sequential(nn.Flatten(2), nn.Transpose(1, 2))
        self.inp_former = INP_Former(
            encoder=inp_encoder,
            bottleneck=nn.ModuleList([MockTransformerBlock(feature_dim)]),
            aggregation=nn.ModuleList([MockTransformerBlock(feature_dim)]),
            decoder=nn.ModuleList([MockTransformerBlock(feature_dim)]),
            prototype_token=nn.Parameter(torch.randn(1, 1, feature_dim)),
            target_layers=[0], fuse_layer_encoder=[[0]], fuse_layer_decoder=[[0]]
        )
        
        # 2. CPR Setup
        cpr_backbone_shapes = [(1, feature_dim, grid_size, grid_size)]
        cpr_backbone = IdentityBackbone(cpr_backbone_shapes)
        lrb = LocalRetrievalBranch([feature_dim], [feature_dim])
        self.cpr = CPR(cpr_backbone, lrb)
        
    def forward(self, feature_map):
        # INP-Former branch tries to reconstruct the feature map
        _, reconstructed_map = self.inp_former(feature_map)
        
        # CPR branch retrieves "normal" features based on the input
        retrieved_maps, _ = self.cpr(feature_map)
        retrieved_map = retrieved_maps[0]

        # Anomaly is the difference between reconstructed and retrieved "normal" features
        anomaly_map = torch.mean((reconstructed_map - retrieved_map)**2, dim=1, keepdim=True)
        return anomaly_map

def generate_anomaly_map(spatial_features: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Runs the hybrid model on a spatial feature grid to generate an anomaly map."""
    grid_h, grid_w, feature_dim = spatial_features.shape
    model = HybridAnomalyModel(feature_dim=feature_dim, grid_size=grid_h)
    model.eval()
    
    # Convert numpy (H, W, C) to torch tensor (1, C, H, W)
    features_tensor = torch.from_numpy(spatial_features).permute(2, 0, 1).unsqueeze(0).float()
    
    with torch.no_grad():
        # Get the low-resolution anomaly map from the model
        low_res_map = model(features_tensor).squeeze().cpu().numpy()
        
    # Upsample the map to the original image's resolution
    high_res_map = cv2.resize(low_res_map, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Normalize for visualization
    map_min, map_max = high_res_map.min(), high_res_map.max()
    if map_max > map_min:
        high_res_map = (high_res_map - map_min) / (map_max - map_min) * 255
    
    return high_res_map.astype(np.uint8)

