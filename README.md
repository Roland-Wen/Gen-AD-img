# Gen-AD-img
### Pre-processing

```bash
# resize raw accident photos to 512×512 PNG
python ./src/data/transforms.py

# segmentation txt to mask
python ./src/data/polygon2mask.py
```

### Concept-tuning

```bash
# get n clean highway images from BDD-100k
python ./src/data/get_reg_imgs.py

# fine-tune with dreambooth
scripts/train_dreambooth.sh
```