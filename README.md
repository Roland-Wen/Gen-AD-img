# Gen-AD-img
### Pre-processing

```bash
# resize raw accident photos to 512×512 PNG
python ./src/data/transforms.py

# segmentation txt to mask
python ./src/data/polygon2mask.py
