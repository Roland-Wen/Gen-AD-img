from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def poly_txt_to_mask(txt_path: Path, out_png: Path, size=512):
    """
    Parse a YOLO-style polygon txt line:
        class xc1 yc1 xc2 yc2 ...  (coords ∈ [0,1])
    and paint it into a single-channel mask.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    with open(txt_path) as fh:
        for line in fh:
            nums = [float(t) for t in line.strip().split()]
            cls, pts = int(nums[0]), nums[1:]
            # convert normalised coords --> pixel tuples
            xy = [(pts[i]  * size, pts[i+1] * size) for i in range(0, len(pts), 2)]
            draw.polygon(xy, fill=cls+1)  # +1 so 0 stays background

    mask.save(out_png, "PNG")

if __name__ == "__main__":
    import glob, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--txt-dir", default="data/raw/accident")
    p.add_argument("--out-dir", default="data/raw/accident_masks")
    p.add_argument("--size",    type=int, default=512)
    a = p.parse_args()

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for txt in glob.glob(f"{a.txt_dir}/*.txt"):
        poly_txt_to_mask(Path(txt),
                         Path(a.out_dir) / (Path(txt).stem + "_mask.png"),
                         a.size)
