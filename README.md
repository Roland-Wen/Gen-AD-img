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

### ControlNet training
```bash
# preprocess cityscape imgs/masks
python ./src/data/conv_and_resz.py
python ./src/utils/gen_controlnet_json.py

# pretrain on cityscape
scripts/pretrain_controlNet.sh

# fine-tune on crash images
scripts/finetune_controlNet.sh
```

### Bulk generation
```bash
python ./src/pipelines/generate_bulk.py
```