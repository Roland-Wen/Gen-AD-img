#!/usr/bin/env python3
"""
generate_bulk.py  –  batch-synthesize crash scenes with ControlNet-Seg

Example
-------
python src/pipelines/generate_bulk.py \
    --dreambooth  models/checkpoints/dreambooth_accident_v1 \
    --controlnet  models/controlnet_accident_seg \
    --mask_dir    data/processed/accident/conditioning_image \
    --prompt_file data/prompts.txt \
    --out_dir     results \
    --n_per_mask  100 \
    --steps       30 \
    --guidance    7.5 \
    --seed        42
"""
import argparse, os, glob, itertools, random, json, time
from pathlib import Path

import torch, tqdm
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline,
                       ControlNetModel, DDIMScheduler,
                       EulerAncestralDiscreteScheduler)

# ---------- helpers ---------------------------------------------------------
def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np_seed = seed % (2**32 - 1)
    try:
        import numpy as np
        np.random.seed(np_seed)
    except ModuleNotFoundError:
        pass

# ---------- main CLI --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dreambooth", default="models/checkpoints/dreambooth_accident_v1", help="Path or hub-id of subject-tuned SD")
    ap.add_argument("--controlnet", default="models/controlnet_accident_seg", help="Path or hub-id of fine-tuned ControlNet")
    ap.add_argument("--mask_dir",   default="data/processed/accident/conditioning_image", help="Folder with *.png masks")
    ap.add_argument("--prompt_file", default="data/prompts.txt", help="Text file: one prompt per line")
    ap.add_argument("--out_dir",    default="results", help="Root folder to save images & masks")
    ap.add_argument("--n_per_mask", type=int, default=100, help="# images to sample per mask")
    ap.add_argument("--steps",      type=int, default=30,  help="DDIM sampling steps")
    ap.add_argument("--guidance",   type=float, default=7.5, help="Classifier-free guidance scale")
    ap.add_argument("--seed",       type=int, default=42,    help="Global seed (0 = random)")
    args = ap.parse_args()

    # ---------- folders -----------------------------------------------------
    out_dir_path = Path(args.out_dir)
    img_out = out_dir_path / "images"; img_out.mkdir(parents=True, exist_ok=True)
    msk_out = out_dir_path / "masks";  msk_out.mkdir(parents=True, exist_ok=True)

    # ---------- checkpointing -----------------------------------------------
    checkpoint_file = out_dir_path / "checkpoint.json"
    checkpoint_data = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            print(f"Resuming from checkpoint: {checkpoint_file}")
        except json.JSONDecodeError:
            print(f"Warning: Checkpoint file {checkpoint_file} is corrupted. Starting fresh.")
            checkpoint_data = {} # Start fresh if checkpoint is invalid
    else:
        print("No checkpoint file found. Starting fresh.")

    # ---------- reproducibility --------------------------------------------
    if args.seed != 0:
        seed_everything(args.seed)

    # ---------- load models (fp16 / CUDA) -----------------------------------
    controlnet = ControlNetModel.from_pretrained(args.controlnet,
                                                 torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.dreambooth,
        controlnet=controlnet,
        # safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda")

    # ---------- data --------------------------------------------------------
    prompts = load_prompts(args.prompt_file)
    masks   = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    if not masks:
        raise ValueError(f"No masks found in {args.mask_dir}")
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    # ---------- generation loop --------------------------------------------
    total_masks = len(masks)
    for i, m_path in enumerate(masks):
        mask_path_obj = Path(m_path)
        mask_name = mask_path_obj.stem # Get filename without extension
        
        print(f"\nProcessing mask {i+1}/{total_masks}: {mask_name}")

        # Get the number of images already generated for this mask from checkpoint
        start_idx = checkpoint_data.get(mask_name, 0)

        if start_idx >= args.n_per_mask:
            print(f"Mask {mask_name} already has {start_idx}/{args.n_per_mask} images. Skipping.")
            continue
        
        cond = Image.open(m_path).convert("L") # Load conditioning image (mask)

        # Use tqdm for the inner loop (per mask)
        for idx in tqdm.tqdm(range(start_idx, args.n_per_mask), 
                             initial=start_idx, 
                             total=args.n_per_mask,
                             desc=f"Generating for {mask_name}",
                             unit="image"):
            if args.seed == 0:                           # fresh seed each frame
                generator.manual_seed(random.randint(0, 2**31))
            else:
                generator.manual_seed(idx)              # deterministic

            current_prompt = prompts[idx % len(prompts)]
            
            try:
                img = pipe(current_prompt,
                           image=cond,
                           num_inference_steps=args.steps,
                           guidance_scale=args.guidance,
                           generator=generator).images[0]

                stem = f"{mask_name}_{idx:06d}.png"
                img.save(img_out / stem)
                cond.save(msk_out / stem)

                # Update checkpoint: record that image 'idx' is done, so 'idx + 1' images are completed.
                checkpoint_data[mask_name] = idx + 1
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)
            
            except Exception as e:
                print(f"\nError generating image {idx} for mask {mask_name} with prompt '{current_prompt}': {e}")
                print("Continuing with the next image or mask...")
                # The checkpoint will not be updated for this failed image, so it will be retried next time.
                break # Break from inner loop (current mask) and proceed to next mask

    # ---------- tiny manifest for traceability ------------------------------
    meta_file = out_dir_path / "meta.json"
    try:
        args_dict = vars(args)
        args_dict['last_run_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2)
        print(f"\nGeneration complete. Manifest saved to {meta_file}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

if __name__ == "__main__":
    main()