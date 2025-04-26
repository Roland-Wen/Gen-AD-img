# Gen-AD-img
Gen-AD-img/

├── README.md     

├── .gitignore

├── data/              

│   ├── raw/                 ← original few-shot accident images, seg-maps

│   ├── processed/           ← resized/cleaned datasets ready for training

│   └── external/            ← links or symlinks to public datasets (Cityscapes…)

├── models/                  ← large binary artefacts (LoRA, DreamBooth, ControlNet)

│   ├── checkpoints/         ← .ckpt or .safetensors files

│   └── configs/             ← lightning / diffusers configs, .yaml

├── src/                     

│   ├── __init__.py

│   ├── data/                ← dataset + dataloader classes

│   │   └── transforms.py

│   ├── training/            ← Lightning modules, loss definitions

│   ├── pipelines/           ← generation pipelines (text2img, seg2img)

│   ├── fine_tune/           ← DreamBooth / LoRA fine-tuning utilities

│   ├── utils/               ← logging, config, metrics, viz helpers

│   └── cli.py               ← `accidentaugment train …`, `accidentaugment gen …`

├── experiments/             ← 1 sub-folder per run; hydra will write here

│   ├── exp-000_loRA/        ← cfg snapshot, tensorboard, stdout

│   └── exp-001_controlnet/

├── notebooks/               ← lightweight EDA & qualitative inspection

│   ├── 01_dataset_preview.ipynb

│   ├── 02_finetune_logs.ipynb

│   └── 03_eval_plots.ipynb

├── scripts/                 ← bash or slurm launchers, quick helpers

│   ├── prepare_data.sh

│   ├── train_lora.sh

│   └── generate_batch.sh

├── tests/                   ← unit + smoke tests (pytest)

│   └── test_dataset.py

├── docs/                    ← project report / slides / diagrams

└── results/                 

    ├── images/              ← sample generations (.png)
    
    └── metrics/             ← mIoU, AP tables (.csv, .json)
