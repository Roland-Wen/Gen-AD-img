import json, glob, os

root = "data/processed/accident" # or /accident
images = sorted(glob.glob(os.path.join(root, "images", "*.png")))

with open(os.path.join(root, "metadata.jsonl"), "w") as f:
    for img_path in images:
        stem = os.path.basename(img_path)
        mask_path = os.path.join("conditioning_image", stem)
        record = {"file_name": os.path.join("images", stem),
                  "conditioning_image_file_name": mask_path,
                  "label": ""}                              # empty caption
        f.write(json.dumps(record) + "\n")
