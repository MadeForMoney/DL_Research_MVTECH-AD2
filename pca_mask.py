# import numpy as np
# from sklearn.decomposition import PCA

# def pca_foreground_mask(image, n_components=1, tau=1.0):
#     X = image.reshape(-1, image.shape[-1])
#     pca = PCA(n_components)
#     pc1 = pca.fit_transform(X)[:, 0].reshape(image.shape[:2])
#     mask = (pc1 > tau).astype(np.uint8)
#     fg = image[mask == 1]
#     bg = image[mask == 0]
#     if np.var(fg) < np.var(bg):
#         mask = 1 - mask
#     return mask

# File: pca_mask.py

import numpy as np
from sklearn.decomposition import PCA

def pca_foreground_mask(image: np.ndarray, n_components: int = 1, tau: float = 1.0) -> np.ndarray:
    """
    Compute a foreground mask using PCA.

    Args:
        image (np.ndarray): Input RGB image (H, W, C).
        n_components (int): Number of PCA components (default: 1).
        tau (float): Threshold applied to the first principal component to define foreground.

    Returns:
        mask (np.ndarray): Binary mask (H, W), 1 = foreground, 0 = background.
    """
    H, W, C = image.shape

    # Flatten spatial dimensions for PCA
    X = image.reshape(-1, C)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(X)[:, 0]  # First principal component
    pc_image = pc.reshape(H, W)

    # Threshold to create mask
    mask = (pc_image > tau).astype(np.uint8)

    # Optional: invert mask if background variance is higher than foreground
    fg_pixels = image[mask == 1]
    bg_pixels = image[mask == 0]

    if fg_pixels.size > 0 and bg_pixels.size > 0:
        if np.var(fg_pixels) < np.var(bg_pixels):
            mask = 1 - mask

    return mask

