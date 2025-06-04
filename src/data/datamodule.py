import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class ContrastiveFragmentDataset(Dataset):
    """
    Each index i ∈ [0..N_images-1] returns a 3‐tuple:
      - patches_16:   FloatTensor of shape (16, 3, 16, 16)
      - positions_16: LongTensor of shape (16,) with values in [0..15]
      - image_id:     LongTensor of shape (1,) containing the integer `i`
    
    Internally, this class expects to be given three file‐paths (strings):
        fragments_path:    path to "fragments_<split>.npy"  (shape: [M_total_fragments, 16, 16, 3])
        ids_path:          path to "ids_<split>.npy"        (shape: [M_total_fragments], each ∈ [0..(N_images-1)])
        positions_path:    path to "positions_<split>.npy"  (shape: [M_total_fragments], each ∈ [0..15])

    On __init__, it does:
      self.fragments = np.load(fragments_path)   # (M_total_fragments, 16, 16, 3)
      self.ids       = np.load(ids_path)         # (M_total_fragments,)
      self.positions = np.load(positions_path)   # (M_total_fragments,)
      self.num_images = int(self.ids.max() + 1)  # # of unique images

    Then it builds a mapping:
      idx_to_patch_indices[i] = array of 16 indices (into the fragments array) that
                                belong to image i.

    Finally, __getitem__(i) picks exactly those 16 patches and returns:
      (patches_tensor, positions_tensor, image_id_tensor).
    """
    def __init__(self, fragments_path: str, ids_path: str, positions_path: str):
        # Expect fragments_path, ids_path, positions_path all to be file‐path strings
        self.fragments = np.load(fragments_path)      # shape: (M_total_fragments, 16, 16, 3)
        self.ids       = np.load(ids_path)            # shape: (M_total_fragments,)
        self.positions = np.load(positions_path)      # shape: (M_total_fragments,)
        self.num_images = int(self.ids.max() + 1)     # suppose ids ∈ [0..N_images-1]

        # Build a mapping: image_id → list of exactly 16 patch‐indices
        self.idx_to_patch_indices = []
        for i in range(self.num_images):
            mask = np.where(self.ids == i)[0]  # array of length 16
            assert len(mask) == 16, f"Expected exactly 16 fragments for image {i}, but got {len(mask)}"
            self.idx_to_patch_indices.append(mask)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        """
        Returns:
            patches_16:   torch.FloatTensor of shape (16, 3, 16, 16)
            positions_16: torch.LongTensor of shape (16,) with values in [0..15]
            image_id:     torch.LongTensor of shape (1,) == idx
        """
        patch_idxs = self.idx_to_patch_indices[idx]  # length 16
        patches_np = self.fragments[patch_idxs]      # shape (16, 16, 16, 3)
        positions  = self.positions[patch_idxs]      # shape (16,)

        # Convert to tensor and permute channels: (16, H, W, 3) → (16, 3, H, W)
        patches_t = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float()  # (16, 3, 16, 16)
        positions_t = torch.from_numpy(positions).long()                     # (16,)
        image_id = torch.tensor(idx, dtype=torch.long)                       # (1,)

        return patches_t, positions_t, image_id


class ContrastiveFragmentDataModule(pl.LightningDataModule):
    """
    Loads fragments_<split>.npy, ids_<split>.npy, positions_<split>.npy via three string paths,
    and returns DataLoaders of batches:

      Train batch: (patches, positions, image_ids)
        - patches      : Tensor of shape (B, 16, 3, 16, 16)
        - positions    : Tensor of shape (B, 16)
        - image_ids    : Tensor of shape (B, 1)

      Val/Test batch: same format, but shuffle=False.

    To use:
        dm = ContrastiveFragmentDataModule(cfg)
        dm.setup()   # builds self.train_dataset, self.val_dataset, self.test_dataset
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
        test_loader  = dm.test_dataloader()
    """
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

        # Read some hyperparameters from config
        self.batch_size  = config["datamodule"]["batch_size_images"]
        self.num_workers = config["datamodule"]["num_workers"]
        self.data_dir    = config["data"]["fragments_dir"]

    def setup(self, stage=None):
        # Build the absolute paths to each .npy file for train/val/test
        t_fr   = os.path.join(self.data_dir, self.cfg["data"]["train_fragments"])
        t_id   = os.path.join(self.data_dir, self.cfg["data"]["train_ids"])
        t_pos  = os.path.join(self.data_dir, self.cfg["data"]["train_positions"])

        v_fr   = os.path.join(self.data_dir, self.cfg["data"]["val_fragments"])
        v_id   = os.path.join(self.data_dir, self.cfg["data"]["val_ids"])
        v_pos  = os.path.join(self.data_dir, self.cfg["data"]["val_positions"])

        te_fr  = os.path.join(self.data_dir, self.cfg["data"]["test_fragments"])
        te_id  = os.path.join(self.data_dir, self.cfg["data"]["test_ids"])
        te_pos = os.path.join(self.data_dir, self.cfg["data"]["test_positions"])

        # ────────────────
        # Create each Dataset by passing the three string paths
        # ────────────────
        self.train_dataset = ContrastiveFragmentDataset(
            fragments_path = t_fr,
            ids_path       = t_id,
            positions_path = t_pos
        )

        self.val_dataset = ContrastiveFragmentDataset(
            fragments_path = v_fr,
            ids_path       = v_id,
            positions_path = v_pos
        )

        self.test_dataset = ContrastiveFragmentDataset(
            fragments_path = te_fr,
            ids_path       = te_id,
            positions_path = te_pos
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,    # Important: no shuffling for val/test
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,    # Important: no shuffling for val/test
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def _collate_fn(batch_list):
        """
        If you ever want a custom collate function, here is a template.
        However, because each __getitem__ now returns (patches, positions, image_id),
        the default PyTorch collate will automatically stack them into:

          patches_batch   : Tensor of shape (B, 16, 3, 16, 16)
          positions_batch : Tensor of shape (B, 16)
          image_ids_batch : Tensor of shape (B, 1)

        If you need to flatten `image_ids_batch` to shape (B,) instead of (B,1),
        you can do so here. For now, we simply rely on the default collate, so you
        do not have to pass `collate_fn=_collate_fn` to DataLoader.
        """
        patches = torch.stack([item[0] for item in batch_list], dim=0)    # (B,16,3,16,16)
        positions = torch.stack([item[1] for item in batch_list], dim=0)  # (B,16)
        image_ids = torch.stack([item[2] for item in batch_list], dim=0)  # (B,1)
        return patches, positions, image_ids.squeeze(1)  # we return image_ids as (B,)

