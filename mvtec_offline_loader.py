"""
mvtec_offline_loader.py

Utility to create offline file lists for each class and split.
Example usage:
    python mvtec_offline_loader.py --root ./mvtec_ad_2 --out ./lists

Will create lists/<class>_train.txt, <class>_test.txt, etc.
"""
import argparse
from pathlib import Path
import dataset_utils as utils
import os

def make_lists(root: str, out_dir: str):
    rootp = Path(root)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    classes = utils.list_classes(root)
    if not classes:
        raise RuntimeError(f"No classes found in {root}")

    for cname in classes:
        croot = rootp / cname
        splits = utils.detect_splits_for_class(str(croot))
        for s in splits:
            folder = croot / s
            files = utils.find_image_files(str(folder))
            if not files:
                continue
            txt_path = outp / f"{cname}_{s}.txt"
            with open(txt_path, "w") as fh:
                for f in files:
                    fh.write(f + "\n")
            print(f"Wrote {len(files)} entries to {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./mvtec_ad_2", help="dataset root")
    parser.add_argument("--out", type=str, default="./lists", help="output lists folder")
    args = parser.parse_args()
    make_lists(args.root, args.out)
