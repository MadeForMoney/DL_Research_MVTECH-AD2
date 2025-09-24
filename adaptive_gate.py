# # File: adaptive_gate.py

# import cv2
# import numpy as np
# import joblib
# import os
# from sklearn.linear_model import LogisticRegression

# # Default path to the pretrained GBDT model inside src/models
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gbdt_gate.pkl")

# def create_dummy_gate_model(model_path=MODEL_PATH):
#     """Creates and saves a placeholder model to make the script runnable."""
#     if os.path.exists(model_path):
#         return
#     print("Creating a dummy GBDT gate model for demonstration...")
#     dummy_clf = LogisticRegression()
#     dummy_clf.fit(np.random.rand(10, 4), np.random.randint(0, 2, 10))
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(dummy_clf, model_path)
#     print(f"Dummy model saved to {model_path}")

# def compute_descriptors(image, mask):
#     # Convert PIL Image to NumPy array if needed
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     fg = image.copy()

#     if fg.ndim > 2:
#         fg = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)  # your images are RGB from PIL

#     if mask is None:
#         mask = np.ones_like(fg, dtype=np.uint8)  # treat all pixels as valid
#     elif not isinstance(mask, np.ndarray):
#         mask = np.array(mask)
#     elif mask.ndim > 2:
#         mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

#     fg[mask == 0] = 0
#     edges = cv2.Canny(fg, 100, 200)
#     edge_density = np.sum(edges > 0) / (fg.shape[0] * fg.shape[1])

#     masked_pixels = fg[mask == 1]
#     if masked_pixels.size == 0:
#         return [edge_density, 0, 0, 0]

#     variance = np.var(masked_pixels)
#     hist = np.histogram(masked_pixels, bins=256, range=(1, 255))[0]
#     prob = hist / (hist.sum() + 1e-7)
#     entropy = -np.sum(prob * np.log2(prob + 1e-7))

#     coords = np.argwhere(mask == 1)
#     if coords.size > 0:
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
#         bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
#         compactness = masked_pixels.size / (bbox_area + 1e-7)
#     else:
#         compactness = 0

#     return [edge_density, variance, entropy, compactness]


# def gbdt_gate(features, model_path=MODEL_PATH):
#     """Uses the GBDT model to decide the feature extraction strategy."""
#     create_dummy_gate_model(model_path)
#     clf = joblib.load(model_path)
#     prob = clf.predict_proba([features])[0][1]
#     return 'multi-scale' if prob > 0.5 else 'single-scale'

# import cv2
# import numpy as np
# import joblib
# import os
# from sklearn.ensemble import GradientBoostingClassifier

# # Default path for the GBDT gate model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gbdt_gate.pkl")

# def compute_descriptors(image, mask=None):
#     """Compute complexity descriptors for an image inside the mask."""
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     fg = image.copy()
#     if fg.ndim > 2:
#         fg = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)

#     if mask is None:
#         mask = np.ones_like(fg, dtype=np.uint8)
#     elif not isinstance(mask, np.ndarray):
#         mask = np.array(mask)
#     elif mask.ndim > 2:
#         mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

#     fg[mask == 0] = 0
#     edges = cv2.Canny(fg, 100, 200)
#     edge_density = np.sum(edges > 0) / (fg.shape[0] * fg.shape[1])

#     masked_pixels = fg[mask == 1]
#     if masked_pixels.size == 0:
#         return [edge_density, 0, 0, 0]

#     variance = np.var(masked_pixels)
#     hist = np.histogram(masked_pixels, bins=256, range=(1, 255))[0]
#     prob = hist / (hist.sum() + 1e-7)
#     entropy = -np.sum(prob * np.log2(prob + 1e-7))

#     coords = np.argwhere(mask == 1)
#     if coords.size > 0:
#         y_min, x_min = coords.min(axis=0)
#         y_max, x_max = coords.max(axis=0)
#         bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
#         compactness = masked_pixels.size / (bbox_area + 1e-7)
#     else:
#         compactness = 0

#     return [edge_density, variance, entropy, compactness]

# def train_gbdt_gate(X, y, model_path=MODEL_PATH):
#     """Train GBDT gate to predict single/multi-scale."""
#     clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
#     clf.fit(X, y)
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(clf, model_path)
#     print(f"✅ Trained GBDT gate saved to {model_path}")
#     return clf

# def gbdt_gate(features, model_path=MODEL_PATH):
#     """Predict single-scale or multi-scale using GBDT gate."""
#     if not os.path.exists(model_path):
#         raise RuntimeError(f"GBDT gate model not found at {model_path}. Train it first.")
#     clf = joblib.load(model_path)
#     prob = clf.predict_proba([features])[0][1]
#     return 'multi-scale' if prob > 0.5 else 'single-scale'

# File: adaptive_gate.py

import cv2
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier

# Path to store/load the trained GBDT gate model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gbdt_gate.pkl")


def compute_descriptors(image, mask=None):
    """
    Compute complexity descriptors for an image inside the mask:
    - Edge Density
    - Texture Variance
    - Entropy
    - Compactness / Foreground Coverage
    """
    # Convert PIL Image to NumPy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    fg = image.copy()
    if fg.ndim > 2:
        fg = cv2.cvtColor(fg, cv2.COLOR_RGB2GRAY)

    if mask is None:
        mask = np.ones_like(fg, dtype=np.uint8)
    elif not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    elif mask.ndim > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Mask the image
    fg[mask == 0] = 0

    # Edge Density
    edges = cv2.Canny(fg, 100, 200)
    edge_density = np.sum(edges > 0) / (fg.shape[0] * fg.shape[1])

    # Texture Variance
    masked_pixels = fg[mask == 1]
    variance = np.var(masked_pixels) if masked_pixels.size > 0 else 0

    # Entropy
    hist = np.histogram(masked_pixels, bins=256, range=(1, 255))[0] if masked_pixels.size > 0 else np.zeros(256)
    prob = hist / (hist.sum() + 1e-7)
    entropy = -np.sum(prob * np.log2(prob + 1e-7))

    # Compactness / Foreground Coverage
    coords = np.argwhere(mask == 1)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        compactness = masked_pixels.size / (bbox_area + 1e-7)
    else:
        compactness = 0

    return [edge_density, variance, entropy, compactness]


def train_gbdt_gate(X, y, model_path=MODEL_PATH):
    """Train GBDT gate model (Gradient Boosted Decision Tree) to decide single/multi-scale."""
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"✅ Trained GBDT gate saved to {model_path}")
    return clf


def gbdt_gate(features, model_path=MODEL_PATH):
    """Predict whether to use single-scale or multi-scale extraction."""
    if not os.path.exists(model_path):
        raise RuntimeError(f"GBDT gate model not found at {model_path}. Train it first.")
    clf = joblib.load(model_path)
    prob = clf.predict_proba([features])[0][1]
    return 'multi-scale' if prob > 0.5 else 'single-scale'


