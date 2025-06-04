import os
os.environ["PL_DISABLE_TORCHMETRICS"] = "1"
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Ensure project root is on PYTHONPATH
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from models2.contrastive_model import ContrastiveFragmentModel
from src.data.datamodule import ContrastiveFragmentDataModule

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def compute_cluster_metrics(true_ids: np.ndarray, pred_ids: np.ndarray):
    """
    Compute ARI and Purity over all patches in test set.
    (Copied verbatim from your original train.py :contentReference[oaicite:2]{index=2}.)
    - true_ids, pred_ids: 1D arrays of length M_test_patches
       where true_ids[i] ∈ [0..N_test_images−1], pred_ids[i] ∈ [0..K−1].
    """
    ari = adjusted_rand_score(true_ids, pred_ids)

    N = len(true_ids)
    clusters = np.unique(pred_ids)
    total_correct = 0
    for c in clusters:
        mask = (pred_ids == c)
        true_in_cluster = true_ids[mask]
        if len(true_in_cluster) == 0:
            continue
        vals, counts = np.unique(true_in_cluster, return_counts=True)
        total_correct += counts.max()
    purity = total_correct / N
    return ari, purity

def reconstruct_groups(encoder, loader, split: str, cfg):
    """
    encoder:      the PatchEncoder module
    loader:       DataLoader for either 'val' or 'test'
    split:        either "val" or "test" (so we know which *_ids.npy to load)
    cfg:          your config dict

    Returns:
      - true_ids        (M_patches,)   the ground-truth image-ID of each patch
      - pred_cluster_ids (M_patches,)  k-means cluster label for each patch
      - embeds_np        (M_patches, D) concatenated embeddings
    """
    assert split in ("val", "test"), "split must be 'val' or 'test'"
    encoder.eval()
    device = next(encoder.parameters()).device

    all_embeddings = []
    with torch.no_grad():
        for batch in loader:
            # Each batch is (patches, positions, image_ids)
            patches, _, _ = batch
            B = patches.shape[0]     # number of images in minibatch
            P = B * 16               # 16 patches per image
            patches_flat = patches.view(P, 3, 16, 16).to(device)
            embeds = encoder(patches_flat)    # shape: (P, D)
            all_embeddings.append(embeds.cpu().numpy())

    all_embeds_np = np.vstack(all_embeddings)  # (M_patches, D)

    # Load the correct ground-truth IDs (.npy) depending on split
    data_dir = cfg["data"]["fragments_dir"]
    if split == "val":
        true_ids = np.load(os.path.join(data_dir, cfg["data"]["val_ids"]))
    else:  # split == "test"
        true_ids = np.load(os.path.join(data_dir, cfg["data"]["test_ids"]))

    assert all_embeds_np.shape[0] == true_ids.shape[0], (
        f"Mismatch: embeddings length {all_embeds_np.shape[0]} vs true_ids length {true_ids.shape[0]}"
    )

    K = int(np.unique(true_ids).max() + 1)
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(all_embeds_np)

    return true_ids, kmeans.labels_, all_embeds_np

def visualize_clusters_after_test(cfg, num_samples: int = 3):
    """
    1) Build val DataLoader, find & load best .ckpt, run reconstruct_groups on val.
    2) Load ground‐truth val_frags, val_ids, val_positions from disk.
    3) For `num_samples` random image‐IDs, reassemble 64×64 "Original" and plot a 4×4 "cluster heatmap".
    """

    device = "cuda" if cfg["trainer"]["gpus"] > 0 and torch.cuda.is_available() else "cpu"

    # Prepare validation DataLoader
    data_module = ContrastiveFragmentDataModule(cfg)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    # Recursively find the latest .ckpt under lightning_logs/contrastive_model/*/checkpoints/
    checkpoint_dir = cfg["checkpoint"]["dirpath"]
    all_ckpts = []
    for root, _, files in os.walk(checkpoint_dir):
        for fname in files:
            if fname.endswith(".ckpt"):
                all_ckpts.append(os.path.join(root, fname))
    if not all_ckpts:
        raise FileNotFoundError(f"No .ckpt found under {checkpoint_dir}")
    ckpt_path = max(all_ckpts, key=os.path.getmtime)
    print(f"→ Loading checkpoint: {ckpt_path}")

    # Load model and switch to eval
    model = ContrastiveFragmentModel.load_from_checkpoint(ckpt_path, hparams=cfg)
    model = model.to(device)
    model.eval()

    # D) Run reconstruct_groups on validation to get (raw_val_ids, pred_val_clusters, _)
    raw_val_ids, pred_val_clusters, _ = reconstruct_groups(
        model.encoder,
        val_loader,
        split="val",
        cfg=cfg
    )

    # Load ground‐truth fragments, IDs, positions from disk
    data_dir  = cfg["data"]["fragments_dir"]
    val_frags = np.load(os.path.join(data_dir, cfg["data"]["val_fragments"]))   # (M_val, 16,16,3)
    val_ids   = np.load(os.path.join(data_dir, cfg["data"]["val_ids"]))         # (M_val,)
    val_pos   = np.load(os.path.join(data_dir, cfg["data"]["val_positions"]))   # (M_val,)

    # Sample some ground‐truth image IDs
    unique_gt_ids   = np.unique(val_ids)
    sampled_images = random.sample(list(unique_gt_ids), min(num_samples, len(unique_gt_ids)))
    os.makedirs("viz_clusters", exist_ok=True)

    cmap = plt.get_cmap("tab10")
    for img_idx in sampled_images:
        # (1) Which 16 patches correspond to this image?
        patch_indices = np.where(val_ids == img_idx)[0]   # shape (16,)
        patches_16    = val_frags[patch_indices]          # (16,16,16,3)
        positions_16  = val_pos[patch_indices]            # (16,)
        clusters_16   = pred_val_clusters[patch_indices]  # (16,)

        # (2) Reassemble the 64×64 “Original” from those 16 patches
        reassembled = np.zeros((64, 64, 3), dtype=patches_16.dtype)
        for k_patch in range(16):
            pos = int(positions_16[k_patch])
            r, c = divmod(pos, 4)
            reassembled[r*16:(r+1)*16, c*16:(c+1)*16] = patches_16[k_patch]

        # (3) Build a 4×4 “local cluster index” array
        unique_clusters = np.unique(clusters_16)
        local_map       = {glob: idx for idx, glob in enumerate(unique_clusters)}
        local_ids       = np.zeros((4, 4), dtype=int)
        for k_patch in range(16):
            glob_label = clusters_16[k_patch]
            local_label = local_map[glob_label]
            pos = int(positions_16[k_patch])
            r, c = divmod(pos, 4)
            local_ids[r, c] = local_label

        # (4) Create a 64×64×3 “heatmap_img” from local_ids
        heatmap_img = np.zeros((64, 64, 3), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                li = local_ids[r, c]
                color_rgb = cmap(li % 10)[:3]
                heatmap_img[r*16:(r+1)*16, c*16:(c+1)*16] = color_rgb

        # (5) Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(np.clip(reassembled, 0, 1))
        axes[0].set_title(f"Image {img_idx}: Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap_img)
        axes[1].set_title(f"Image {img_idx}: Cluster Heatmap")
        axes[1].axis("off")

        # (6) Annotate each 16×16 block with its local cluster index
        for r in range(4):
            for c in range(4):
                li = local_ids[r, c]
                y = r * 16 + 8
                x = c * 16 + 8
                axes[1].text(
                    x, 64 - y,
                    str(li),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold"
                )

        plt.suptitle(f"Clusters for Image {img_idx}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join("viz_clusters", f"image_{img_idx}_cluster.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    print(f"Saved cluster visuals for {sampled_images} → viz_clusters/")


def plot_loss_curves(model):
    """
    Read these per‐epoch lists from the LightningModule:
      - model.train_loss_total_epochs
      - model.val_loss_total_epochs
      - model.train_loss_infonce_epochs
      - model.train_loss_slot_epochs

    Then produce two PNGs:
      1) loss_curves_total.png      (Train vs. Val Total Loss)
      2) loss_curves_components.png (Train InfoNCE vs. Train Slot‐CE vs. Train Total)
    """

    # (A) Gather epoch‐level values
    train_total    = model.train_loss_total_epochs    # list of length T_train
    val_total      = model.val_loss_total_epochs      # list of length T_val (often T_train+1)
    train_infonce  = model.train_loss_infonce_epochs  # typically length = T_train
    train_slot_ce  = model.train_loss_slot_epochs     # typically length = T_train

    # (B) Determine the number of epochs to plot (common prefix)
    n_epochs = min(len(train_total), len(val_total))
    epochs = np.arange(1, n_epochs + 1)

    # Truncate each series to the first n_epochs entries:
    train_total    = train_total[:n_epochs]
    val_total      = val_total[:n_epochs]
    train_infonce  = train_infonce[:n_epochs]
    train_slot_ce  = train_slot_ce[:n_epochs]

    # ─── Plot 1: Train vs Val Total Loss ────────────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_total,    marker='o', label='Train Total Loss')
    plt.plot(epochs, val_total,      marker='s', label='Val Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Train vs. Val Total Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves_total.png", dpi=150)
    plt.close()

    # ─── Plot 2: Train InfoNCE, Train Slot‐CE, Train Total ─────────────────────
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_infonce,  marker='o', label='Train InfoNCE')
    plt.plot(epochs, train_slot_ce,  marker='s', label='Train Slot‐CE')
    plt.plot(epochs, train_total,    marker='^', label='Train Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Component Losses")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves_components.png", dpi=150)
    plt.close()

def main(config_path: str):
    # 1) Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2) Seed + device
    pl.seed_everything(42)
    gpus = cfg["trainer"].get("gpus", 0)
    device = torch.device("cuda") if (gpus > 0 and torch.cuda.is_available()) else torch.device("cpu")

    # 3) DataModule
    data_module = ContrastiveFragmentDataModule(cfg)
    data_module.setup()

    # 4) Model & callbacks
    model = ContrastiveFragmentModel(hparams=cfg)
    tb_logger = TensorBoardLogger(save_dir=cfg["checkpoint"]["dirpath"], name="tensorboard_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg["checkpoint"]["monitor"],
        mode=cfg["checkpoint"]["mode"],
        dirpath=cfg["checkpoint"]["dirpath"],
        filename=cfg["checkpoint"]["filename"],
        save_top_k=cfg["checkpoint"]["save_top_k"],
    )
    earlystop_callback = EarlyStopping(
        monitor=cfg["checkpoint"]["monitor"],
        mode=cfg["checkpoint"]["mode"],
        patience=cfg["checkpoint"].get("patience", 5),
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        precision=cfg["trainer"]["precision"],
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        deterministic=True,
    )

    # 5) Fit
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    print("\n=== Generating loss curves ===")
    plot_loss_curves(model)
    print("Saved loss_curves_total.png and loss_curves_components.png\n")

    # 6) Test + print Test ARI & Purity (same as before)
    print("=== Evaluating on test set ===")
    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt or not os.path.isfile(best_ckpt):
        print("ERROR: No checkpoint found; skipping test evaluation.")
    else:
        best_model = ContrastiveFragmentModel.load_from_checkpoint(best_ckpt, hparams=cfg).to(device)
        raw_test_ids, pred_test_clusters, _ = reconstruct_groups(
            best_model.encoder, data_module.test_dataloader(), split="test", cfg=cfg
        )
        ari, purity = compute_cluster_metrics(raw_test_ids, pred_test_clusters)
        print(f"Test ARI:    {ari:.4f}")
        print(f"Test Purity: {purity:.4f}\n")

    # 7) Visualize a few sample cluster assignments
    print("=== Visualizing sample cluster assignments ===")
    num_samples = cfg.get("num_visual_samples", 3)
    visualize_clusters_after_test(cfg, num_samples=num_samples)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train + Evaluate + Plot + Visualize")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
