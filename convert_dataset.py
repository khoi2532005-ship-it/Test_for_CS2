import json, shutil, random
from pathlib import Path
from PIL import Image

# === CONFIG ===
DATASET_ROOT  = "E:\\archive"
OUTPUT_DIR    = "E:\\dataset"
SPLIT         = (0.7, 0.2, 0.1)
USE_BG_IMAGES = True

species_folders = sorted([f for f in Path(DATASET_ROOT).iterdir() if f.is_dir()])
class_names = []
all_samples = []  # (image_path, bbox, class_idx)

for class_idx, species_dir in enumerate(species_folders):
    species_name = species_dir.name.split("-", 1)[-1].strip()
    class_names.append(species_name)

    img_folder = species_dir / ("images with background" if USE_BG_IMAGES else "images without backgrounds")
    coco_dir   = species_dir / "annotations_coco"

    # load all bboxes for this species
    bboxes = []
    for ann_file in coco_dir.glob("*.json"):
        with open(ann_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            for ann in data:
                if "bbox" in ann:
                    bboxes.append(ann["bbox"])

    if not bboxes:
        continue

    # get all images in the folder
    images = list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))

    # pair each image with bboxes (one bbox per image, cycling if needed)
    for i, img_path in enumerate(images):
        bbox = bboxes[i % len(bboxes)]
        all_samples.append((img_path, bbox, class_idx))

random.shuffle(all_samples)

n       = len(all_samples)
n_train = int(n * SPLIT[0])
n_val   = int(n * SPLIT[1])

splits = {
    "train": all_samples[:n_train],
    "val":   all_samples[n_train:n_train + n_val],
    "test":  all_samples[n_train + n_val:]
}

# clear old output
shutil.rmtree(Path(OUTPUT_DIR) / "images", ignore_errors=True)
shutil.rmtree(Path(OUTPUT_DIR) / "labels", ignore_errors=True)

for split_name, samples in splits.items():
    img_out = Path(OUTPUT_DIR) / "images" / split_name
    lbl_out = Path(OUTPUT_DIR) / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, bbox, class_idx in samples:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except:
            continue

        x, y, w, h = bbox
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H
        cx, cy, nw, nh = [max(0.0, min(1.0, v)) for v in [cx, cy, nw, nh]]

        dst_img = img_out / img_path.name
        shutil.copy(img_path, dst_img)

        lbl_file = lbl_out / (img_path.stem + ".txt")
        with open(lbl_file, "w") as f:
            f.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

yaml_content = f"""path: E:\\dataset
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
