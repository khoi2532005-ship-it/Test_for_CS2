import json, os, shutil
from pathlib import Path

# === CONFIG ===
DATASET_ROOT = "E:\\archive"   # folder with 01-Dromaius... subfolders
OUTPUT_DIR = "E:\\dataset"
SPLIT = (0.7, 0.2, 0.1)  # train / val / test
USE_BG_IMAGES = True  # True = images with background, False = without

# collect all species folders
species_folders = sorted([f for f in Path(DATASET_ROOT).iterdir() if f.is_dir()])
class_names = []
all_samples = []  # list of (image_path, bbox, class_idx)

for class_idx, species_dir in enumerate(species_folders):
    species_name = species_dir.name.split("-", 1)[-1].strip()  # e.g. "Dromaius novaehollandiae"
    class_names.append(species_name)

    img_folder = species_dir / ("images with background" if USE_BG_IMAGES else "images without backgrounds")
    coco_file = species_dir / "annotations_coco"

    # find all coco json files
    for ann_file in coco_file.glob("*.json"):
        with open(ann_file) as f:
            annotations = json.load(f)
        if not isinstance(annotations, list):
            annotations = [annotations]

        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, w, h]
            image_id = ann["image_id"]

            # find matching image file
            matches = list(img_folder.glob(f"*{image_id}*"))
            if not matches:
                matches = list(img_folder.glob("*.jpg"))  # fallback

            for img_path in matches:
                all_samples.append((img_path, bbox, class_idx))
                break

import random
random.shuffle(all_samples)

n = len(all_samples)
n_train = int(n * SPLIT[0])
n_val = int(n * SPLIT[1])

splits = {
    "train": all_samples[:n_train],
    "val": all_samples[n_train:n_train + n_val],
    "test": all_samples[n_train + n_val:]
}

from PIL import Image

for split_name, samples in splits.items():
    img_out = Path(OUTPUT_DIR) / "images" / split_name
    lbl_out = Path(OUTPUT_DIR) / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, bbox, class_idx in samples:
        # get image dimensions for normalization
        try:
            with Image.open(img_path) as img:
                W, H = img.size
        except:
            continue

        x, y, w, h = bbox
        # convert COCO [x,y,w,h] → YOLO [cx,cy,w,h] normalized
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H

        # clamp to [0,1]
        cx, cy, nw, nh = [max(0, min(1, v)) for v in [cx, cy, nw, nh]]

        # copy image
        dst_img = img_out / img_path.name
        shutil.copy(img_path, dst_img)

        # write label
        lbl_file = lbl_out / (img_path.stem + ".txt")
        with open(lbl_file, "w") as f:
            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# write dataset.yaml
yaml_content = f"""path: {Path(OUTPUT_DIR).resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
with open(Path(OUTPUT_DIR) / "dataset.yaml", "w") as f:
    f.write(yaml_content)

print(f"Done! {n} samples → train:{n_train} val:{n_val} test:{n - n_train - n_val}")
print(f"Classes: {len(class_names)}")
