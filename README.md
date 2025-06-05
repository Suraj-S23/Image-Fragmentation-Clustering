# Image-Fragmentation-Clustering
Image Fragmentation & Contrastive Clustering
This project trains a small neural network to split 64×64 images into 16 patches (16×16 each), learn patch-level embeddings via a contrastive (InfoNCE) loss, and predict each patch’s position in a 4×4 grid (jigsaw or “slot” loss). After training, patch embeddings are clustered to recover which fragments came from the same image, and a separate head predicts each patch’s original slot.

Folder & File Overview
config.yaml
Global configuration (paths, batch size, learning rate, loss weights, etc.).

data_fragmentation.py
Reads raw ImageNet-64 batches and splits each 64×64 image into 16 non-overlapping patches.
Outputs .npy files for fragments, their image IDs, and their grid positions.

datamodule.py
Wraps the .npy fragment files into PyTorch datasets and provides train/val/test data loaders.

contrastive_model.py
Defines the neural network:

A lightweight CNN that maps each 16×16 patch to a D-dim embedding (normalized).

A small MLP head that predicts which of the 16 grid slots a patch belongs to.

eval_metrics.py
Compute clustering metrics (Adjusted Rand Index and Purity) between true and predicted patch-to-image assignments.

train.py

Runs the full training loop (InfoNCE + slot losses), validation, and testing.

After training, it runs k-means on patch embeddings (to recover image groups) and reports ARI/Purity.

Also plots loss curves and saves a few “original vs. cluster” visualization PNGs.

inference.py
(Optional) A standalone script for evaluators:

Fragments one batch of validation images.

Loads a saved checkpoint, runs clustering and slot predictions, prints ARI/Purity/slot-accuracy, and saves sample visualizations.

Quick Usage
Prepare data (ImageNet-64 pickles) and fill out config.yaml with correct paths.

Install dependencies:

bash
Copy
Edit
pip install torch torchvision pytorch-lightning numpy scikit-learn pillow matplotlib pyyaml
Train & Validate:

bash
Copy
Edit
python train.py --config config.yaml
This will train the model, evaluate on validation/test, plot loss curves, and generate a few cluster visuals under viz_clusters/.

Inference (metrics + visuals):

bash
Copy
Edit
python inference.py \
  --data_dir path/to/imagenet64_batches \
  --ckpt   lightning_logs/contrastive_model/version_<k>/checkpoints/best.ckpt \
  --num_vis 3
This will fragment one validation batch, compute ARI/Purity and slot-accuracy, and save side-by-side visualizations (ground-truth vs. cluster heatmap) under viz_clusters/.

Core Idea in Brief
Fragmentation: Each 64×64 image → sixteen 16×16 patches.

Embedding: A small CNN encodes each patch into a D-dim vector.

Losses:

InfoNCE: Pull patches from the same image together, push others apart.

Slot Head: Predict each patch’s grid location (1 of 16).

Clustering: After training, run k-means on all patch embeddings to group patches by original image; evaluate with ARI/Purity.

Visualization: Show a few 64×64 ground-truth reconstructions alongside a “cluster heatmap” illustrating which patches were grouped together.
