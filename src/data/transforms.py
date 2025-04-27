from pathlib import Path
from typing import Iterable
from PIL import Image

def resize_to_square(
    in_paths: Iterable[Path],
    out_dir: Path,
    size: int = 512,
    keep_aspect: bool = False,
    suffix: str = ".png",
) -> None:
    """
    Resize images to `size`×`size` and save as PNG.

    Args:
        in_paths  : iterable of image file Paths
        out_dir   : destination directory (created if absent)
        size      : target height and width in pixels
        keep_aspect: if True, adds letterbox padding instead of stretching
        suffix    : output file extension
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in in_paths:
        img = Image.open(p).convert("RGB")
        if keep_aspect:
            img.thumbnail((size, size), Image.LANCZOS)
            pad = Image.new("RGB", (size, size), (0, 0, 0))
            pad.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
            img = pad
        else:
            img = img.resize((size, size), Image.LANCZOS)

        out_path = out_dir / (p.stem + suffix)
        img.save(out_path, format="PNG", optimize=True)


if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Batch-resize accident images")
    parser.add_argument("--input-glob", type=str, default="data/raw/accident/*")
    parser.add_argument("--out-dir",    type=Path, default=Path("data/processed/accident"))
    parser.add_argument("--size",       type=int, default=512)
    parser.add_argument("--keep-aspect", action="store_true")
    args = parser.parse_args()

    files = [Path(p) for p in glob.glob(args.input_glob)]
    resize_to_square(files, args.out_dir, args.size, args.keep_aspect)
