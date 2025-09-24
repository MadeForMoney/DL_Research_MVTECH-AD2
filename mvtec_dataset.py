"""
mvtec_dataset.py
PyTorch Dataset for MVTEC-AD-2 style layout.

Returns:
    image_tensor, mask_tensor or None, label, metadata_dict
"""

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import Optional, List, Callable, Tuple, Dict

# Default transforms (can override with torchvision transforms)
def default_transforms(image_size: int = 256):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

class MVTecAD2Dataset(Dataset):
    def __init__(
        self,
        root: str,
        category: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        load_masks: bool = False,
        image_exts: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        if image_exts is None:
            image_exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".tif"]
        self.image_exts = image_exts
        self.split = split
        self.transform = transform if transform is not None else default_transforms()
        self.load_masks = load_masks

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._build_index(category)

    def _is_image(self, p: Path) -> bool:
        return p.suffix.lower() in self.image_exts

    def _build_index(self, category):
        # Check category folders under root
        classes = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        print(f"DEBUG: Found class folders in root '{self.root}': {[d.name for d in classes]}")

        # Filter for single category if provided
        if category:
            classes = [d for d in classes if d.name == category]
            if not classes:
                raise ValueError(f"Category '{category}' not found under {self.root}")

        for cdir in classes:
            cname = cdir.name
            self.class_to_idx[cname] = len(self.class_to_idx)
            self.idx_to_class[self.class_to_idx[cname]] = cname

            # Check if folder contains duplicate category (like can/can)
            # Robust nested folder detection: if cdir contains a subfolder with the same name, go deeper
            nested_folder = cdir / cname
            if nested_folder.exists() and nested_folder.is_dir():
                cdir = nested_folder
                print(f"DEBUG: Found nested category folder, updated cdir: {cdir}")


            # Find the split folder
            candidate = cdir / self.split
            if not candidate.exists():
                # fallback for 'test_private' / 'test' etc.
                if self.split in ("test", "test_private"):
                    candidate = cdir / "test"
            if not candidate.exists() or not candidate.is_dir():
                print(f"WARNING: Split folder '{self.split}' not found in {cdir}, skipping this class.")
                continue

            # Collect images
            for f in sorted(candidate.iterdir()):
                if f.is_file() and self._is_image(f):
                    mask_p = None
                    if self.load_masks:
                        gt_dirs = ["ground_truth", "gt", "masks", "groundtruth"]
                        for gname in gt_dirs:
                            gp = cdir / gname / f.name
                            if gp.exists():
                                mask_p = gp
                                break
                            gp2 = cdir / gname / (f.stem + ".png")
                            if gp2.exists():
                                mask_p = gp2
                                break
                    self.samples.append((str(f), str(mask_p) if mask_p else None, cname))

        if not self.samples:
            raise RuntimeError(f"No images found for split='{self.split}' under {self.root}. Check dataset layout.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, Dict]:
        img_path, mask_path, cname = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img) if self.transform else T.ToTensor()(img)

        mask_t = None
        if self.load_masks and mask_path is not None:
            mask_p = Path(mask_path)
            if mask_p.exists():
                mask = Image.open(mask_p).convert("L")
                # Resize mask to match image tensor size
                img_h, img_w = img_t.shape[1], img_t.shape[2]
                mask = mask.resize((img_w, img_h), Image.NEAREST)
                mask_bin = (np.array(mask) > 127).astype(np.float32)
                mask_t = torch.from_numpy(mask_bin).unsqueeze(0)

        # Label: 0 = normal, 1 = defective
        label = 0 if ("train" in str(img_path).lower() or "good" in str(img_path).lower()) else 1
        meta = {"img_path": img_path, "mask_path": mask_path, "class": cname}

        return img_t, mask_t, label, meta

# Convenience function to create dataloaders
def make_dataloader(
    root: str,
    category: Optional[str],
    split: str,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    load_masks: bool = False,
):
    ds = MVTecAD2Dataset(root=root, category=category, split=split, transform=transform, load_masks=load_masks)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dl
