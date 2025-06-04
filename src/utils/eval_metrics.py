import os
import sys
import yaml
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Ensure project root is on PYTHONPATH
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from models2.contrastive_model import ContrastiveFragmentModel
from src.data.datamodule import ContrastiveFragmentDataModule
from src.train import reconstruct_groups
import pytorch_lightning as pl
import torch



def compute_cluster_metrics(true_ids, pred_ids):
    ari = adjusted_rand_score(true_ids, pred_ids)
    N = len(true_ids)
    clusters = np.unique(pred_ids)
    total_correct = 0
    for c in clusters:
        mask = (pred_ids == c)
        if not mask.any():
            continue
        true_in_cluster = true_ids[mask]
        vals, counts = np.unique(true_in_cluster, return_counts=True)
        total_correct += counts.max()
    purity = total_correct / N
    return ari, purity


def histogram_fallback(data_dir, cfg):
    frags = np.load(os.path.join(data_dir, cfg["data"]["val_fragments"]))  # (M_val,16,16,3)
    val_ids = np.load(os.path.join(data_dir, cfg["data"]["val_ids"]))      # (M_val,)
    N, H, W, C = frags.shape
    feats = []
    for i in range(N):
        patch = (frags[i] * 255).astype(np.uint8)
        h_r, _ = np.histogram(patch[:, :, 0], bins=4, range=(0, 256))
        h_g, _ = np.histogram(patch[:, :, 1], bins=4, range=(0, 256))
        h_b, _ = np.histogram(patch[:, :, 2], bins=4, range=(0, 256))
        feat = np.concatenate([h_r, h_g, h_b]).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-8
        feats.append(feat / norm)
    feats = np.vstack(feats)
    K = int(np.unique(val_ids).max() + 1)
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(feats)
    return val_ids, kmeans.labels_



def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if cfg["trainer"]["gpus"] > 0 and torch.cuda.is_available() else "cpu"
    pl.seed_everything(42)

    # 1) DataModule (validation only)
    data_module = ContrastiveFragmentDataModule(cfg)
    data_module.setup()

    # 2) Load checkpoint
    checkpoint_dir = cfg["checkpoint"]["dirpath"]
    ckpt_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
    model = ContrastiveFragmentModel.load_from_checkpoint(ckpt_path, hparams=cfg)
    model.eval().to(device)

    # 3) Contrastive clustering on validation
    raw_val_ids, pred_val_clusters, _ = reconstruct_groups(
        model.encoder, data_module.val_dataloader(), cfg
    )
    ari_val, purity_val = compute_cluster_metrics(raw_val_ids, pred_val_clusters)
    print("=== Contrastive (val) Metrics ===")
    print(f"ARI    : {ari_val:.4f}")
    print(f"Purity : {purity_val:.4f}")

    # 4) Optional: histogram fallback
    if cfg["eval"].get("use_histogram_fallback", False):
        true_ids_hist, pred_hist = histogram_fallback(cfg["data"]["fragments_dir"], cfg)
        ari_hist, purity_hist = compute_cluster_metrics(true_ids_hist, pred_hist)
        print("\n=== Histogram‐only (val) Metrics ===")
        print(f"ARI    : {ari_hist:.4f}")
        print(f"Purity : {purity_hist:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Model on Validation")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
