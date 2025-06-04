import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric


class ResBlock(nn.Module):
    """
    A single 2‐layer residual block for 16×16 patches → embed_dim features.
    2 blocks build a lightweight encoder
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class PatchEncoder(nn.Module):
    """
    Encoder for individual 16×16 patches → embedding of size embed_dim (unit norm).
    We use:
      Conv(3→32) → ResBlock(32→64, stride=2) → ResBlock(64→128, stride=2) → pool → FC(embed_dim).
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)  # 32×16×16
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResBlock(32, 64, stride=2)   # (64×8×8)
        self.layer2 = ResBlock(64, 128, stride=2)  # (128×4×4)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # (128×1×1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))  # (P,32,16,16)
        out = self.layer1(out)                    # (P,64,8,8)
        out = self.layer2(out)                    # (P,128,4,4)
        out = self.adaptive_pool(out)             # (P,128,1,1)
        out = self.flatten(out)                   # (P,128)
        embed = F.normalize(self.fc(out), dim=1)  # (P,embed_dim), unit‐norm
        return embed


class ContrastiveFragmentModel(pl.LightningModule):
    """
    LightningModule combining InfoNCE (contrastive) + absolute‐slot prediction (CE over 16 slots).
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        cfg_model = self.hparams["model"]
        D = cfg_model["embed_dim"]  

        # Patch encoder
        self.encoder = PatchEncoder(embed_dim=D)

        # Slot‐prediction head: three‐layer MLP → 16 logits (more capacity)
        self.pos_head = nn.Sequential(
            nn.Linear(D, D),           # 64 → 64
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(D, D // 2),      # 64 → 32
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(D // 2, 16),     # 32 → 16 slots
        )
        self.pos_loss_fxn = nn.CrossEntropyLoss()

        # Hyperparameters
        self.lr = cfg_model["lr"]
        self.temperature = cfg_model["temperature"]
        self.jigsaw_weight = cfg_model["jigsaw_loss_weight"]

        # Trackers
        self.train_loss_infonce = MeanMetric()
        self.train_loss_slot = MeanMetric()
        self.train_loss_total = MeanMetric()

        self.val_loss_infonce = MeanMetric()
        self.val_loss_slot = MeanMetric()
        self.val_loss_total = MeanMetric()

        # Similarity trackers (for plotting)
        self.train_pos_sims = []
        self.train_neg_sims = []
        self.val_pos_sims = []
        self.val_neg_sims = []

        # Per‐epoch lists
        self.train_loss_infonce_epochs = []
        self.train_loss_slot_epochs = []
        self.train_loss_total_epochs = []
        self.val_loss_infonce_epochs = []
        self.val_loss_slot_epochs = []
        self.val_loss_total_epochs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def info_nce_loss(self, embeddings: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (P, D) where P = B×16
        image_ids:  (P,) long tensor, each in [0..B-1]
        """
        P, D = embeddings.shape
        tau = self.temperature

        # Cosine similarity matrix: (P,P)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / tau  # (P,P)
        exp_sim = torch.exp(sim_matrix)

        # positive mask (image_ids[i] == image_ids[j]) minus self‐pairs
        eye = torch.eye(P, device=embeddings.device).bool()
        img_eq = image_ids.unsqueeze(1) == image_ids.unsqueeze(0)
        same_mask = img_eq & (~eye)       # (P,P)

        # Compute numerator & denominator
        numerator = (exp_sim * same_mask.float()).sum(dim=1)
        denom = exp_sim.sum(dim=1) - exp_sim.diagonal()

        # Avoid log(0)
        eps = 1e-8
        numerator = torch.clamp(numerator, min=eps)
        denom = torch.clamp(denom, min=eps)

        loss = -torch.log(numerator / denom).mean()
        return loss

    # Shared step for train/val
    def _shared_step(self, batch, batch_idx, stage: str):
        """
        batch = (patches, positions, image_ids)
        patches:   (B, 16, 3, 16, 16)
        positions: (B, 16)
        image_ids: (B,)
        """
        patches, positions, image_ids = batch
        B, P, C, H, W = patches.shape       # P should be 16 (patches)
        device = patches.device

        # Flatten patches for the encoder
        patches_flat = patches.view(B * P, C, H, W)  # (B*P, 3,16,16)

        # Compute embeddings
        embeds = self.encoder(patches_flat)  # (B*P, D)

        # Slot‐CE loss
        positions_flat = positions.view(-1).to(device)  # (B*P,)
        slot_logits    = self.pos_head(embeds)           # (B*P, num_slots)
        loss_slot      = self.pos_loss_fxn(slot_logits, positions_flat)

        # InfoNCE (contrastive) loss
        image_ids_flat = image_ids.unsqueeze(1).repeat(1, P).view(-1).to(device)  # (B*P,)

        # Normalized cosine‐similarity matrix
        embeds_norm = F.normalize(embeds, dim=1)                 # (B*P, D)
        sims = torch.matmul(embeds_norm, embeds_norm.t()) / self.temperature  # (B*P, B*P)
        exp_sims = torch.exp(sims)

        # Build “same‐image but not self” mask
        eye      = torch.eye(B * P, device=device).bool()
        img_eq   = (image_ids_flat.unsqueeze(1) == image_ids_flat.unsqueeze(0))
        mask_pos = img_eq & (~eye)         # True where same image, i != j
        mask_neg = (~eye)                  # True everywhere except diagonal

        # Compute numerator & denominator
        numerator = (exp_sims * mask_pos.float()).sum(dim=1)          # (B*P,)
        denom     = exp_sims.sum(dim=1) - exp_sims.diagonal()         # (B*P,)

        eps = 1e-8
        numerator = torch.clamp(numerator, min=eps)
        denom     = torch.clamp(denom, min=eps)

        loss_infonce = -torch.log(numerator / denom).mean()

        # Total loss
        total_loss = loss_infonce + self.jigsaw_weight * loss_slot # jigsaw weight still selected by trial-and-error

        # Update MeanMetric trackers
        if stage == "train":
            self.train_loss_infonce.update(loss_infonce)
            self.train_loss_slot.update(loss_slot)
            self.train_loss_total.update(total_loss)
        else:
            self.val_loss_infonce.update(loss_infonce)
            self.val_loss_slot.update(loss_slot)
            self.val_loss_total.update(total_loss)

        # Compute & store per‐batch avg pos/neg sims for epoch logging
        avg_pos = (sims * mask_pos.float()).sum() / mask_pos.float().sum()
        avg_neg = (sims * mask_neg.float()).sum() / mask_neg.float().sum()

        if stage == "train":
            if not hasattr(self, "_epoch_train_pos_sims"):
                self._epoch_train_pos_sims = []
                self._epoch_train_neg_sims = []
            self._epoch_train_pos_sims.append(avg_pos.detach().cpu())
            self._epoch_train_neg_sims.append(avg_neg.detach().cpu())
        else:
            if not hasattr(self, "_epoch_val_pos_sims"):
                self._epoch_val_pos_sims = []
                self._epoch_val_neg_sims = []
            self._epoch_val_pos_sims.append(avg_pos.detach().cpu())
            self._epoch_val_neg_sims.append(avg_neg.detach().cpu())

        # Log to TensorBoard/CSV via Lightning
        self.log(f"{stage}/infonce",    loss_infonce, prog_bar=(stage=="train"), on_epoch=True)
        self.log(f"{stage}/slot_ce",    loss_slot,    prog_bar=(stage=="train"), on_epoch=True)
        self.log(f"{stage}/total_loss", total_loss,   prog_bar=(stage=="train"), on_epoch=True)

        return total_loss
        
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def on_train_epoch_end(self):
        # Epoch‐level pos/neg sim computation for train
        if hasattr(self, "_epoch_train_pos_sims"):
            epos = float(torch.tensor(self._epoch_train_pos_sims).mean())
            eneg = float(torch.tensor(self._epoch_train_neg_sims).mean())
            self.train_pos_sims.append(epos)
            self.train_neg_sims.append(eneg)
            self._epoch_train_pos_sims.clear()
            self._epoch_train_neg_sims.clear()

        # Record epoch losses
        self.train_loss_infonce_epochs.append(self.train_loss_infonce.compute().item())
        self.train_loss_slot_epochs.append(self.train_loss_slot.compute().item())
        self.train_loss_total_epochs.append(self.train_loss_total.compute().item())
        self.train_loss_infonce.reset()
        self.train_loss_slot.reset()
        self.train_loss_total.reset()

    def on_validation_epoch_end(self):
        if hasattr(self, "_epoch_val_pos_sims"):
            epos = float(torch.tensor(self._epoch_val_pos_sims).mean())
            eneg = float(torch.tensor(self._epoch_val_neg_sims).mean())
            self.val_pos_sims.append(epos)
            self.val_neg_sims.append(eneg)
            self._epoch_val_pos_sims.clear()
            self._epoch_val_neg_sims.clear()

        self.val_loss_infonce_epochs.append(self.val_loss_infonce.compute().item())
        self.val_loss_slot_epochs.append(self.val_loss_slot.compute().item())
        self.val_loss_total_epochs.append(self.val_loss_total.compute().item())
        self.val_loss_infonce.reset()
        self.val_loss_slot.reset()
        self.val_loss_total.reset()

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4  # small weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams["trainer"]["max_epochs"],
            eta_min=self.lr * 0.01
        )
        return [optimizer], [scheduler]
