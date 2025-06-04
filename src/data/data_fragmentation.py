import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import argparse
import tempfile
import logging
import yaml
import json
import numpy as np
import pickle
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import Imagenet64  


tmpdir = tempfile.gettempdir()
os.environ["TMPDIR"] = tmpdir  # Ensures Imagenet64 won’t crash if it needs a temp directory

def fragment_images(images: np.ndarray, grid_size: int) -> np.ndarray:
    """
    images: np.ndarray of shape (N, H, W, 3), where H = W and H % grid_size == 0.
    grid_size: int, the number of patches along each axis (e.g. 4 → 4×4 grid).
    Returns: np.ndarray of shape (N * grid_size^2, patch_h, patch_w, 3).
    """
    N, H, W, C = images.shape
    assert H == W, "Images must be square"
    assert H % grid_size == 0, f"Image size {H} not divisible by grid_size {grid_size}"
    patch_size = H // grid_size

    # Reshape to (N, grid_size, patch_size, grid_size, patch_size, C)
    x = images.reshape(N, grid_size, patch_size, grid_size, patch_size, C)
    # Transpose to (N, grid_size, grid_size, patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Flatten to (N * grid_size^2, patch_size, patch_size, C)
    patches = x.reshape(-1, patch_size, patch_size, C)
    return patches

def process_split(
    split_name: str,
    split_indices: np.ndarray,
    all_images: np.ndarray,
    grid_size: int,
    chunk_size_full: int,
    output_dir: Path,
    seed: int,
):
    """
    split_name: "train", "val", or "test"
    split_indices: 1D array of image‐indices (referring to rows in all_images)
    all_images: np.ndarray of shape (num_samples, 64, 64, 3)
    grid_size: how many patches along each axis
    chunk_size_full: how many full images to fragment at once - for memmory concerns
    output_dir: Path to a folder where we write three .npy files:
        - fragments_<split>.npy
        - ids_<split>.npy
        - positions_<split>.npy
    seed: for reproducible shuffling of patches
    """
    rng = np.random.RandomState(
        seed + (0 if split_name == "train" else (1 if split_name == "val" else 2))
    )

    num_images = len(split_indices)
    patch_per_image = grid_size * grid_size
    total_patches = num_images * patch_per_image
    patch_size = all_images.shape[1] // grid_size  

    frag_shape = (total_patches, patch_size, patch_size, 3)
    id_shape = (total_patches,)

    frag_path = output_dir / f"fragments_{split_name}.npy"
    ids_path = output_dir / f"ids_{split_name}.npy"
    pos_path = output_dir / f"positions_{split_name}.npy"

    fragments_mmap = np.lib.format.open_memmap(
        frag_path,
        mode="w+",
        dtype=np.float32,
        shape=frag_shape,
    )
    ids_mmap = np.lib.format.open_memmap(
        ids_path,
        mode="w+",
        dtype=np.int32,
        shape=id_shape,
    )
    positions_mmap = np.lib.format.open_memmap(
        pos_path,
        mode="w+",
        dtype=np.int32,
        shape=id_shape,
    )

    offset = 0
    idx_array = np.array(split_indices)
    num_chunks = int(np.ceil(num_images / chunk_size_full))

    logging.info(
        f"[{split_name}] Total images: {num_images}, "
        f"chunks: {num_chunks} (chunk size = {chunk_size_full})"
    )

    for chunk_i in range(num_chunks):
        start_img = chunk_i * chunk_size_full
        end_img = min((chunk_i + 1) * chunk_size_full, num_images)
        chunk_inds = idx_array[start_img:end_img]
        chunk_images = all_images[chunk_inds]  # shape (n_chunk, 64, 64, 3)

        patches = fragment_images(chunk_images, grid_size)
        n_patches = patches.shape[0]  # total num patches - should equal (end_img-start_img)*patch_per_image

        # Create local image‐IDs within this split
        local_ids = np.arange(start_img, end_img, dtype=np.int32)
        local_ids = np.repeat(local_ids, patch_per_image)

        # Create “position index” 0..(patch_per_image-1) for each patch of each image
        #     So each image in [start_img..end_img) contributes [0,1,2,…,patch_per_image-1].
        single_positions = np.arange(patch_per_image, dtype=np.int32)  # [0..15] for a 4×4 grid (16 patches)
        local_positions = np.tile(single_positions, end_img - start_img)

        # Shuffle patches, local_ids, local_positions in lockstep to not lose alignment
        perm = rng.permutation(n_patches)
        patches = patches[perm]
        local_ids = local_ids[perm]
        local_positions = local_positions[perm]

        # Write them into the memmaps at [offset : offset + n_patches]
        fragments_mmap[offset : offset + n_patches, :, :, :] = patches
        ids_mmap[offset : offset + n_patches] = local_ids
        positions_mmap[offset : offset + n_patches] = local_positions
        offset += n_patches

        logging.info(
            f"[{split_name}] Wrote chunk {chunk_i+1}/{num_chunks} → "
            f"patches {offset-n_patches}:{offset}"
        )

    # Full‐split shuffle
    logging.info(f"[{split_name}] Performing full‐split shuffle of {total_patches} patches")
    idx_full = np.arange(total_patches)
    rng.shuffle(idx_full)

    fragments_all = fragments_mmap[:]  
    ids_all = ids_mmap[:]
    positions_all = positions_mmap[:]

    fragments_mmap[:] = fragments_all[idx_full]
    ids_mmap[:] = ids_all[idx_full]
    positions_mmap[:] = positions_all[idx_full]

    logging.info(
        f"[{split_name}] Done. Files on disk:\n"
        f"  {frag_path}\n  {ids_path}\n  {pos_path}"
    )

    # Clean up
    del fragments_mmap, ids_mmap, positions_mmap
    del fragments_all, ids_all, positions_all
    gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fragment a set of Imagenet64 images into (G×G) patches "
                    "and save train/val/test splits, plus visual confirmation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML or JSON config file. If omitted, looks for 'config.yaml' next to this script.",
        default="config/frag_config.yaml",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to the directory containing the Imagenet64 pickled batches.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save fragments, ids, positions, and confirmation visuals.")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Total number of full 64×64 images to sample.")
    parser.add_argument("--batch-size-full", type=int, default=None,
                        help="How many full 64×64 images to load in RAM at once.")
    parser.add_argument("--grid-size", type=int, choices=[2, 4, 8], default=None,
                        help="Split each image into (grid-size × grid-size) patches.")
    parser.add_argument("--test-size", type=float, default=None,
                        help="Fraction of num_samples to hold out as test.")
    parser.add_argument("--val-size", type=float, default=None,
                        help="Fraction of remaining (train+val) to hold out as val.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--chunk-size-full", type=int, default=None,
                        help="When writing each split, how many full images to fragment at once.")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, print DEBUG logs; otherwise INFO.")

    args = parser.parse_args()

    # Determine script directory
    script_dir = Path(__file__).resolve().parent

    # If no --config is passed, look for config.yaml or config.json in script directory
    if args.config is None:
        yaml_path = script_dir / "config.yaml"
        json_path = script_dir / "config.json"
        if yaml_path.exists():
            args.config = yaml_path
        elif json_path.exists():
            args.config = json_path

    # Load config file if provided
    if args.config:
        if args.config.suffix in (".yaml", ".yml"):
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        elif args.config.suffix == ".json":
            with open(args.config, 'r') as f:
                cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {args.config.suffix}")

        # Populate args from cfg if they aren't already set via CLI
        for key, val in cfg.items():
            if hasattr(args, key) and getattr(args, key) is None:
                if key in ("data_dir", "output_dir"):
                    setattr(args, key, Path(val))
                else:
                    setattr(args, key, val)

    # Verify all required args are set
    required_attrs = [
        "data_dir", "output_dir", "num_samples", "batch_size_full",
        "grid_size", "test_size", "val_size", "seed", "chunk_size_full"
    ]
    missing = [attr for attr in required_attrs if getattr(args, attr) is None]
    if missing:
        raise ValueError(f"Missing required arguments or config entries: {missing}")

    return args

def save_visual_confirmation(full_images: np.ndarray, grid_size: int, save_dir: Path):
    """
    Creates three figures:
      1. Four original 64×64 images with a grid overlay.
      2. The G×G fragments of the first image (row-major).
      3. A random selection of 16 patches from the pool of fragments of those 4 images.
    Saves each figure as a PNG under save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have at least 4 images
    num_available = full_images.shape[0]
    if num_available < 4:
        n = num_available
    else:
        n = 4
    sample_imgs = full_images[:n]  # shape (n, 64, 64, 3)

    # Plot original images with grid overlay
    fig1, axs1 = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        img = sample_imgs[i]
        axs1[i].imshow(img)
        axs1[i].axis('off')
        patch_size = img.shape[0] // grid_size
        for j in range(1, grid_size):
            axs1[i].axhline(j * patch_size, color='white', linestyle='--', linewidth=1)
            axs1[i].axvline(j * patch_size, color='white', linestyle='--', linewidth=1)
        axs1[i].set_title(f"Image {i}")
    fig1.suptitle("Original 64×64 Images with Grid Overlay", y=1.02)
    fig1.tight_layout()
    fig1_path = save_dir / "originals_with_grid.png"
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # Fragment the first image and plot its fragments
    first_img = sample_imgs[:1]  # shape (1, 64, 64, 3)
    patches_img0 = fragment_images(first_img, grid_size)  # shape (grid_size^2, patch_h, patch_h, 3)
    fig2, axs2 = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx, patch in enumerate(patches_img0):
        r = idx // grid_size
        c = idx % grid_size
        axs2[r, c].imshow(patch)
        axs2[r, c].axis('off')
    fig2.suptitle("Fragments of Image 0 (Row-major Order)", y=1.02)
    fig2.tight_layout()
    fig2_path = save_dir / "fragments_image0.png"
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    # Take fragments from all n images, shuffle, show random G^2 patches
    all_patches = fragment_images(sample_imgs, grid_size)  # shape (n * grid_size^2, patch_h, patch_h, 3)
    total_patches = all_patches.shape[0]
    num_to_show = min(grid_size * grid_size, total_patches)
    np.random.seed(42)
    selected_indices = np.random.choice(total_patches, num_to_show, replace=False)
    selected_patches = all_patches[selected_indices]

    fig3, axs3 = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx in range(grid_size * grid_size):
        r = idx // grid_size
        c = idx % grid_size
        if idx < num_to_show:
            axs3[r, c].imshow(selected_patches[idx])
        axs3[r, c].axis('off')
    fig3.suptitle("Random Patches from Unordered Pool", y=1.02)
    fig3.tight_layout()
    fig3_path = save_dir / "random_patches.png"
    fig3.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig3)

def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    np.random.seed(args.seed)

    # Memory-safe loading: temporarily patch os.listdir so Imagenet64 sees only one batch - memory concerns
    logging.info(f"Safely loading up to {args.num_samples} images from a single batch in {args.data_dir} (seed={args.seed})")
    original_listdir = os.listdir
    os.listdir = lambda path: ["train_data_batch_1"] if "train_data" in str(path) else original_listdir(path)
    try:
        dataset = Imagenet64(args.data_dir)
        gen = dataset.datagen_cls(batch_size=args.num_samples, ds="train", augmentation=False)
        samples, _ = next(gen)            # Loads up to num_samples images from that single batch
        full_images = samples.numpy()     # NumPy array of shape (num_samples, 64, 64, 3)
    finally:
        os.listdir = original_listdir
        del dataset, gen
        gc.collect()

    logging.info(f"Safely loaded {full_images.shape[0]} images. (dtype={full_images.dtype}, range=[{full_images.min():.3f},{full_images.max():.3f}])")

    # Create output subfolder (with key params in name)
    run_name = f"fragments_g{args.grid_size}_n{args.num_samples}_s{args.seed}"
    run_out = args.output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving everything into output folder: {run_out}")

    # Visualization confirmation
    vis_dir = run_out / "frag_confirmation"
    logging.info(f"Generating visual confirmation in {vis_dir}")
    save_visual_confirmation(full_images, args.grid_size, vis_dir)

    # Split into train/val/test by image index
    all_indices = np.arange(full_images.shape[0])
    trainval_idx, test_idx = train_test_split(
        all_indices, test_size=args.test_size, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=args.val_size / (1 - args.test_size), random_state=args.seed
    )
    logging.info(f"Split sizes (in #images): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Process each split
    for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        process_split(
            split_name=split_name,
            split_indices=idxs,
            all_images=full_images,
            grid_size=args.grid_size,
            chunk_size_full=args.chunk_size_full,
            output_dir=run_out,
            seed=args.seed,
        )

    logging.info("All splits completed successfully.")


if __name__ == "__main__":
    main()
