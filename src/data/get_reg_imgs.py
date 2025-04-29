import os
import json
import argparse
from pathlib import Path
from typing import Tuple
from PIL import Image

def filter_and_resize_bdd_images(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    target_scene: str,
    target_size: Tuple[int, int],
    max_files: int = float('inf'), # Default to processing all found files
    verbose: bool = False
) -> None:
    """
    Filters BDD100k images based on scene attribute, resizes them, and saves them.

    Args:
        labels_dir  : Path to the directory containing BDD100k JSON label files.
        images_dir  : Path to the directory containing corresponding BDD100k JPG image files.
        output_dir  : Path to the destination directory (created if absent).
        target_scene: The scene attribute value to filter for (e.g., 'highway').
        target_size : Target (width, height) tuple in pixels for resizing.
        max_files   : Maximum number of images to process and save. Defaults to infinity.
        verbose     : If True, print detailed processing information for each file.
    """
    # --- Create output directory if it doesn't exist ---
    output_dir.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured output directory '{output_dir}' exists.")

    # --- Process files ---
    processed_label_count = 0
    copied_count = 0

    # Check if the labels directory exists
    if not labels_dir.is_dir():
        print(f"Error: Labels directory not found at '{labels_dir}'. Please check the path.")
        return # Exit the function if labels dir is missing

    # Check if the images directory exists
    if not images_dir.is_dir():
        print(f"Error: Images directory not found at '{images_dir}'. Please check the path.")
        return # Exit the function if images dir is missing

    print(f"Scanning '{labels_dir}' for JSON files...")

    try:
        # Iterate through files using Path.iterdir() for better cross-platform compatibility
        json_files = sorted(list(labels_dir.glob('*.json'))) # Get a sorted list for potentially more deterministic runs
        total_json_files = len(json_files)
        print(f"Found {total_json_files} JSON files.")

        for json_path in json_files:
            if copied_count >= max_files:
                print(f"\nReached the limit of {max_files} files. Stopping processing.")
                break

            processed_label_count += 1
            base_name = json_path.stem # Gets 'xxxx' from 'xxxx.json'
            image_filename = base_name + '.jpg'
            image_path = images_dir / image_filename
            # Save with original image filename but in the output directory
            destination_path = output_dir / image_filename

            if verbose:
                print(f"\nProcessing label {processed_label_count}/{total_json_files}: {json_path.name}")

            try:
                # Read and parse the JSON file
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Access the scene attribute safely using .get()
                scene = data.get('attributes', {}).get('scene')

                if scene == target_scene:
                    if verbose:
                        print(f"  Scene matches '{target_scene}'. Looking for image: {image_filename}")

                    # Check if the corresponding image file exists
                    if image_path.is_file():
                        if verbose:
                            print(f"  Found corresponding image: {image_path}. Resizing and copying...")
                        try:
                            # Open, resize, and save the image
                            # Use LANCZOS for high-quality downsampling (matches the example)
                            # Newer Pillow versions prefer Image.Resampling.LANCZOS
                            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

                            img = Image.open(image_path).convert("RGB") # Ensure RGB
                            resized_img = img.resize(target_size, resample_filter)
                            resized_img.save(destination_path, format="JPEG", quality=95) # Save as JPEG

                            if verbose:
                                print(f"  Successfully saved resized image to {destination_path}")
                            copied_count += 1

                        except FileNotFoundError: # Should be caught by is_file, but belt-and-suspenders
                            print(f"  Error: Image file disappeared unexpectedly: {image_path}")
                        except IOError as e:
                            print(f"  Error processing image {image_filename}: {e}")
                        except Exception as e:
                            print(f"  An unexpected error occurred with image {image_filename}: {e}")
                    else:
                         if verbose:
                             print(f"  Warning: Corresponding image not found at {image_path}")
                # else: # Optional: uncomment to see all skipped files
                #     if verbose:
                #         print(f"  Skipping: scene is '{scene}' (not '{target_scene}')")

            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file: {json_path}")
            except Exception as e:
                # Catch any other unexpected errors during JSON processing
                print(f"An unexpected error occurred while processing {json_path.name}: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during directory scanning or processing: {e}")

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Total JSON labels checked in '{labels_dir}': {processed_label_count}")
    print(f"Images matching '{target_scene}' copied and resized to '{output_dir}': {copied_count}")


# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter BDD100k images by scene, resize, and save them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Required arguments
    parser.add_argument("--labels-dir", type=Path, default="data/raw/bdd100k_labels_100k/100k/train",
                        help="Directory containing BDD100k JSON label files.")
    parser.add_argument("--images-dir", type=Path, default="data/raw/bdd100k_images_100k/100k/train",
                        help="Directory containing corresponding BDD100k JPG image files.")
    parser.add_argument("--output-dir", type=Path, default="data/regularization",
                        help="Directory to save the processed images.")
    parser.add_argument("--target-scene", type=str, default='highway',
                        help="Scene attribute value to filter for (e.g., 'highway', 'city street').")

    # Optional arguments with defaults
    parser.add_argument("--width", type=int, default=512,
                        help="Target width in pixels for resizing.")
    parser.add_argument("--height", type=int, default=512,
                        help="Target height in pixels for resizing.")
    parser.add_argument("--max-files", "-n", type=int, default=300,
                        help="Maximum number of images to process and save. Set to 0 or negative for no limit.")
    parser.add_argument("--verbose", "-v", action="store_false",
                        help="Print detailed information during processing.")

    args = parser.parse_args()

    # Handle max_files interpretation (argparse defaults to float('inf'), but user might enter <= 0)
    max_files_limit = args.max_files if args.max_files > 0 else float('inf')

    # Construct the target size tuple
    size_tuple = (args.width, args.height)

    # Call the main function
    filter_and_resize_bdd_images(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        target_scene=args.target_scene,
        target_size=size_tuple,
        max_files=max_files_limit,
        verbose=args.verbose
    )