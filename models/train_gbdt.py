# print("Starting")

# import os
# import joblib
# import numpy as np
# from tqdm import tqdm
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report, accuracy_score

# from src.mvtec_dataset import MVTecAD2Dataset
# from src.adaptive_gate import compute_descriptors

# print("Imports done")


# def extract_features(root_path):
#     """
#     Extract features from the exact folder path.
#     """
#     from src.mvtec_dataset import MVTecAD2Dataset
#     from src.adaptive_gate import compute_descriptors

#     # Here root_path points directly to the folder containing images
#     dataset = MVTecAD2Dataset(
#         root=root_path,
#         category=None,   # already specific folder
#         split="all",     # grab everything inside
#         transform=None,
#         load_masks=False
#     )

#     features, labels = [], []

#     for img, mask, label, meta in tqdm(dataset, desc=f"Extracting features"):
#         desc = compute_descriptors(img, mask)
#         features.append(desc)
#         labels.append(label)

#     return np.array(features), np.array(labels)


# if __name__ == "__main__":
#     # Hardcoded full path
#     CAN_TRAIN_PATH = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2\can\can\train\good"
#     CAN_validate_PATH  = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2\can\can\validate"

#     print("🔹 Extracting training features for 'can'")
#     X_train, y_train = extract_features(CAN_TRAIN_PATH)

#     print("🔹 Extracting validate features for 'can'")
#     X_validate, y_validate = extract_features(CAN_validate_PATH)

#     # Train GBDT
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.metrics import accuracy_score, classification_report
#     import joblib
#     import os

#     gbdt = GradientBoostingClassifier(
#         n_estimators=200,
#         learning_rate=0.05,
#         max_depth=3,
#         random_state=42
#     )
#     gbdt.fit(X_train, y_train)

#     y_pred = gbdt.predict(X_validate)
#     print(f"✅ Accuracy:", accuracy_score(y_validate, y_pred))
#     print(classification_report(y_validate, y_pred))

#     os.makedirs("models", exist_ok=True)
#     joblib.dump(gbdt, "models/gbdt_model_can.pkl")
#     print("✅ GBDT model saved")


# print("Starting")

# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib
# from PIL import Image
# from src.adaptive_gate import compute_descriptors

# print("Imports done")


# def extract_features_hardcoded(img_folder):
#     """
#     Load images from a folder and extract descriptors.
#     Labels:
#         - folder containing 'good' -> 0 (normal)
#         - otherwise -> 1 (defective)
#     """
#     features, labels = [], []

#     img_folder = os.path.abspath(img_folder)
#     if not os.path.exists(img_folder):
#         raise RuntimeError(f"Folder not found: {img_folder}")

#     for fname in tqdm(os.listdir(img_folder), desc=f"Extracting features from {img_folder}"):
#         fpath = os.path.join(img_folder, fname)
#         if not fpath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
#             continue

#         img = Image.open(fpath).convert("RGB")
#         img_np = np.array(img)  # convert PIL Image to NumPy array
#         desc = compute_descriptors(img_np, mask=None)

#         features.append(desc)
#         labels.append(0 if "good" in img_folder.lower() else 1)

#     return np.array(features), np.array(labels)


# if __name__ == "__main__":
#     # ------------------ HARD-CODED PATHS ------------------
#     DATA_ROOT = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2"

#     CATEGORIES = {
#         "can": {
#             "train": os.path.join(DATA_ROOT, "can", "can", "train", "good"),
#             "validate": os.path.join(DATA_ROOT, "can", "can", "validate")
#         },
#         "fruit_jelly": {
#             "train": os.path.join(DATA_ROOT, "fruit_jelly", "fruit_jelly", "train", "good"),
#             "validate": os.path.join(DATA_ROOT, "fruit_jelly", "fruit_jelly", "validate")
#         },
#         "vial": {
#             "train": os.path.join(DATA_ROOT, "vial", "vial", "train", "good"),
#             "validate": os.path.join(DATA_ROOT, "vial", "vial", "validate")
#         }
#     }

#     for CATEGORY, paths in CATEGORIES.items():
#         print(f"\n🔹 Extracting features for category: {CATEGORY}")
#         X_train, y_train = extract_features_hardcoded(paths["train"])
#         X_validate, y_validate = extract_features_hardcoded(paths["validate"])

#         # ------------------ TRAIN GBDT ------------------
#         gbdt = GradientBoostingClassifier(
#             n_estimators=200,
#             learning_rate=0.05,
#             max_depth=3,
#             random_state=42
#         )
#         gbdt.fit(X_train, y_train)

#         y_pred = gbdt.predict(X_validate)
#         print(f"✅ Accuracy for {CATEGORY}:", accuracy_score(y_validate, y_pred))
#         print(classification_report(y_validate, y_pred))

#         # ------------------ SAVE MODEL ------------------
#         os.makedirs("models", exist_ok=True)
#         model_path = f"models/gbdt_model_{CATEGORY}.pkl"
#         joblib.dump(gbdt, model_path)
#         print(f"✅ GBDT model saved to {model_path}")


# print("Starting")

# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib
# from PIL import Image
# from src.adaptive_gate import compute_descriptors

# print("Imports done")


# def extract_features_hardcoded(img_folder):
#     features, labels = [], []
#     img_folder = os.path.abspath(img_folder)

#     for fname in tqdm(os.listdir(img_folder), desc=f"Extracting features from {img_folder}"):
#         fpath = os.path.join(img_folder, fname)
#         if not fpath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
#             continue

#         img = np.array(Image.open(fpath).convert("RGB"))
#         desc = compute_descriptors(img, mask=None)
#         features.append(desc)
#         labels.append(0)  # all train/good images are normal

#     return np.array(features), np.array(labels)


# if __name__ == "__main__":
#     # Hardcoded paths for 'can' only
#     CAN_TRAIN_PATH = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2\can\can\train\good"
#     CAN_VALIDATE_PATH = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2\can\can\validation\good"

#     print(f"\n🔹 Extracting training features for 'can'")
#     X_train, y_train = extract_features_hardcoded(CAN_TRAIN_PATH)

#     print(f"\n🔹 Extracting validation features for 'can'")
#     X_validate, y_validate = extract_features_hardcoded(CAN_VALIDATE_PATH)

#     # Train GBDT
#     gbdt = GradientBoostingClassifier(
#         n_estimators=200,
#         learning_rate=0.05,
#         max_depth=3,
#         random_state=42
#     )
#     gbdt.fit(X_train, y_train)

#     # Evaluate
#     y_pred = gbdt.predict(X_validate)
#     print(f"✅ Accuracy for 'can':", accuracy_score(y_validate, y_pred))
#     print(classification_report(y_validate, y_pred))

#     # Save model
#     os.makedirs("models", exist_ok=True)
#     model_path = "models/gbdt_model_can.pkl"
#     joblib.dump(gbdt, model_path)
#     print(f"✅ GBDT model saved to {model_path}")


    
#     import random

#     print("\n🔹 Sample predictions on random validation images:")
#     # Select 5 random indices from the validation set
#     random_indices = random.sample(range(len(X_validate)), 5)

#     for i in random_indices:
#         decision = 'multi-scale' if clf.predict([X_validate[i]])[0] == 1 else 'single-scale'
#         print(f"Image {i}: {decision}")



# Working Code

# import os
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from src.adaptive_gate import compute_descriptors, train_gbdt_gate
# from sklearn.metrics import accuracy_score

# def extract_features(img_folder):
#     """Extract complexity descriptors for all images in a folder."""
#     if not os.path.exists(img_folder):
#         raise RuntimeError(f"Folder not found: {img_folder}")

#     features = []
#     image_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

#     for fname in tqdm(image_files, desc=f"Extracting features from {img_folder}"):
#         img_path = os.path.join(img_folder, fname)
#         img = Image.open(img_path).convert("RGB")
#         desc = compute_descriptors(img, mask=None)
#         features.append(desc)

#     return np.array(features)


# if __name__ == "__main__":
#     DATA_ROOT = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2"

#     # Hardcoded category 'can'
#     CAN_TRAIN_PATH = os.path.join(DATA_ROOT, "can", "can", "train", "good")
#     CAN_VALIDATE_PATH = os.path.join(DATA_ROOT, "can", "can", "validation", "good")

#     # 1️⃣ Extract descriptors from training images
#     print("\n🔹 Extracting training features for 'can'")
#     X_train = extract_features(CAN_TRAIN_PATH)

#     # 2️⃣ Extract descriptors from validation images
#     print("\n🔹 Extracting validation features for 'can'")
#     X_validate = extract_features(CAN_VALIDATE_PATH)

#     # 3️⃣ Create heuristic labels for GBDT gate
#     # Simple heuristic: edge_density > mean → multi-scale
#     edge_density_train = X_train[:, 0]
#     threshold_train = np.mean(edge_density_train)
#     y_train = (edge_density_train > threshold_train).astype(int)  # 1 = multi-scale, 0 = single-scale

#     # Apply same threshold to validation set for labels
#     edge_density_validate = X_validate[:, 0]
#     y_validate = (edge_density_validate > threshold_train).astype(int)

#     # 4️⃣ Train the GBDT gate model
#     clf = train_gbdt_gate(X_train, y_train)

#     # 5️⃣ Predict on validation set
#     y_pred = clf.predict(X_validate)
#     accuracy = accuracy_score(y_validate, y_pred)
#     print(f"\n🔹 Validation Accuracy: {accuracy*100:.2f}%")

#     # Optional: print first 5 predictions
#     print("\nSample predictions:")
#     for i in range(min(5, len(X_validate))):
#         decision = 'multi-scale' if y_pred[i] == 1 else 'single-scale'
#         print(f"Image {i}: {decision}, Label: {'multi-scale' if y_validate[i] == 1 else 'single-scale'}")


# File: train_gbdt.py

# File: train_gbdt.py

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.adaptive_gate import compute_descriptors,train_gbdt_gate
from src.pca_mask import pca_foreground_mask  # PCA-based foreground mask

# ---------- Feature Extraction Function ----------
def extract_features(img_folder):
    """Extract complexity descriptors for all images in a folder using PCA mask."""
    if not os.path.exists(img_folder):
        raise RuntimeError(f"Folder not found: {img_folder}")

    features = []
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

    for fname in tqdm(image_files, desc=f"Extracting features from {img_folder}"):
        img_path = os.path.join(img_folder, fname)
        img = np.array(Image.open(img_path).convert("RGB"))

        # Step 1: Get PCA-based foreground mask
        mask = pca_foreground_mask(img, n_components=1, tau=1.0)

        # Step 2: Compute complexity descriptors using mask
        desc = compute_descriptors(img, mask=mask)
        features.append(desc)

    return np.array(features)



# ---------- Main ----------
if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\amnit\Desktop\rank1_codeV2\rank1_codeV2\rank1_code\mvtec_ad_2"

    # Hardcoded category 'can'
    CAN_TRAIN_PATH = os.path.join(DATA_ROOT, "can", "can", "train", "good")
    CAN_VALIDATE_PATH = os.path.join(DATA_ROOT, "can", "can", "validation", "good")

    # 1️⃣ Extract descriptors from training images
    print("\n🔹 Extracting training features for 'can'")
    X_train = extract_features(CAN_TRAIN_PATH)

    # 2️⃣ Extract descriptors from validation images (optional)
    print("\n🔹 Extracting validation features for 'can'")
    X_validate = extract_features(CAN_VALIDATE_PATH)

    # 3️⃣ Create heuristic labels for GBDT gate
    # Heuristic: edge_density (first descriptor) above mean → multi-scale
    edge_density = X_train[:, 0]
    threshold = np.mean(edge_density)
    y_train = (edge_density > threshold).astype(int)
    y_validate = (X_validate[:, 0] > threshold).astype(int)

    # 4️⃣ Train the GBDT gate model
    print("\n🔹 Training GBDT Adaptive Gate...")
    clf = train_gbdt_gate(X_train, y_train)
    print("Training complete!")

    # 5️⃣ Evaluate on validation set
    y_pred = clf.predict(X_validate)
    accuracy = accuracy_score(y_validate, y_pred)
    print(f"\n🔹 Validation Accuracy: {accuracy*100:.2f}%")

    # Optional: print first 5 predictions
    print("\nSample predictions:")
    for i in range(min(13, len(X_validate))):
        decision = 'multi-scale' if y_pred[i] == 1 else 'single-scale'
        label_str = 'multi-scale' if y_validate[i] == 1 else 'single-scale'
        print(f"Image {i}: {decision}, Label: {label_str}")


