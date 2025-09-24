import torch
import torch.nn as nn
import numpy as np
import cv2

# Note: This requires PyTorch Geometric. pip install torch-geometric
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    from skimage.segmentation import slic
except ImportError:
    print("Warning: PyTorch Geometric or scikit-image not found. GNN refiner will not work.")
    GCNConv = None

class GNNRefiner(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16):
        super().__init__()
        if GCNConv is None: raise ImportError("PyTorch Geometric is required for GNNRefiner.")
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).sigmoid()
        return x

def _create_graph_from_mask(mask, n_segments=100):
    # 1. Create superpixels from the mask
    segments = slic(mask, n_segments=n_segments, compactness=10, start_label=1)
    
    # 2. Create nodes (one for each superpixel)
    num_nodes = len(np.unique(segments))
    node_features = []
    for i in np.unique(segments):
        # Feature is the average mask value in that superpixel
        avg_intensity = np.mean(mask[segments == i])
        node_features.append([avg_intensity / 255.0])
    x = torch.tensor(node_features, dtype=torch.float)

    # 3. Create edges between adjacent superpixels
    edges = set()
    h, w = segments.shape
    for i in range(h - 1):
        for j in range(w - 1):
            if segments[i, j] != segments[i + 1, j]:
                edges.add(tuple(sorted((segments[i, j]-1, segments[i + 1, j]-1))))
            if segments[i, j] != segments[i, j + 1]:
                edges.add(tuple(sorted((segments[i, j]-1, segments[i, j + 1]-1))))
    
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index), segments

def gnn_based_refinement(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    if GCNConv is None:
        print("Skipping GNN refinement as dependencies are missing.")
        return mask
        
    model = GNNRefiner()
    model.eval()

    graph_data, segments = _create_graph_from_mask(mask)
    
    with torch.no_grad():
        refined_node_scores = model(graph_data)
    
    # Reconstruct the mask from the refined node scores
    refined_mask = np.zeros_like(mask, dtype=np.uint8)
    for i, score in enumerate(refined_node_scores):
        if score.item() > 0.5:
            refined_mask[segments == i + 1] = 255
            
    return refined_mask

if __name__ == '__main__':
    dummy_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (30, 30), (100, 100), 255, -1)
    dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
    
    if GCNConv:
        refined = gnn_based_refinement(dummy_mask, dummy_image)
        print("GNN-Based Refinement Demo")
        print(f"Refined mask shape: {refined.shape}")
        # cv2.imwrite("gnn_refined_mask.png", refined)
