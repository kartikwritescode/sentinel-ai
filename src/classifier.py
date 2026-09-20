# Production-Grade Multi-Scale Conv-BiLSTM + Multi-Head Temporal Attention Classifier
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Union
import config

SCALER_PATH = "data/scaler_stats.npz"


class FeatureScaler:
    """
    Z-Score Standardization across the feature dimensions.
    Normalizes features to zero-mean and unit variance: (X - mu) / (sigma + eps).
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # X: (N, seq_len, feat_dim)
        flat = X.reshape(-1, X.shape[-1])
        self.mean = np.mean(flat, axis=0).astype(np.float32)
        self.std = (np.std(flat, axis=0) + 1e-6).astype(np.float32)

    def transform(self, X):
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path=SCALER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path=SCALER_PATH):
        if os.path.exists(path):
            data = np.load(path)
            self.mean = data['mean']
            self.std = data['std']
            return True
        return False


class TemporalAugmenter:
    """
    Heavy data augmentation for temporal kinematic sequences:
      1. Gaussian Kinematic Jitter (stronger noise)
      2. Channel Dropout (prevents over-reliance on single features)
      3. Temporal Speed Perturbation (simulates 0.8x-1.2x movement speed)
      4. Random Temporal Shift (rolls sequence by ±3 frames)
    """
    @staticmethod
    def augment(X_batch, jitter_std=0.05, dropout_prob=0.15):
        """Apply non-mixing augmentations to a batch."""
        N, seq_len, feat_dim = X_batch.shape
        augmented = X_batch.clone()

        # 1. Gaussian noise (stronger than before: 0.05 vs 0.02)
        noise = torch.randn_like(augmented) * jitter_std
        augmented = augmented + noise

        # 2. Random Feature Channel Dropout (stronger: 15% vs 8%, 70% probability vs 50%)
        if torch.rand(1).item() < 0.7:
            # Per-sample dropout mask (not batch-level)
            mask = (torch.rand(N, 1, feat_dim, device=X_batch.device) > dropout_prob).float()
            augmented = augmented * mask

        # 3. Temporal Speed Perturbation (resample to simulate 0.8x-1.2x speed)
        if torch.rand(1).item() < 0.5:
            speed_factor = 0.8 + torch.rand(1).item() * 0.4  # uniform [0.8, 1.2]
            new_len = int(seq_len * speed_factor)
            new_len = max(5, min(new_len, seq_len * 2))  # safety bounds
            # Permute to (N, feat_dim, seq_len) for interpolate, then back
            aug_perm = augmented.permute(0, 2, 1)  # (N, feat_dim, seq_len)
            resampled = F.interpolate(aug_perm, size=seq_len, mode='linear', align_corners=False)
            augmented = resampled.permute(0, 2, 1)  # (N, seq_len, feat_dim)

        # 4. Random Temporal Shift (roll by ±3 frames)
        if torch.rand(1).item() < 0.5:
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=1)

        return augmented

    @staticmethod
    def mixup(X_batch, y_batch, alpha=0.4):
        """
        Mixup augmentation: blends pairs of samples and labels.
        Returns mixed features and mixed labels.
        """
        N = X_batch.size(0)
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5

        perm = torch.randperm(N, device=X_batch.device)
        X_mixed = lam * X_batch + (1.0 - lam) * X_batch[perm]
        y_mixed = lam * y_batch + (1.0 - lam) * y_batch[perm]

        return X_mixed, y_mixed


# Model Architecture 


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Dynamically re-weights feature channels per sample, suppressing irrelevant
    features (e.g., pairwise distance when only 1 person is present).
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal self-attention (2 heads).
    Each head independently learns to attend to different phases of the event
    (e.g., approach phase vs. strike phase).
    """
    def __init__(self, feature_dim, num_heads=2):
        super().__init__()
        assert feature_dim % num_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 2),
                nn.Tanh(),
                nn.Linear(self.head_dim // 2, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        B, T, D = x.shape
        x_heads = x.view(B, T, self.num_heads, self.head_dim)

        contexts = []
        for i, head_attn in enumerate(self.attention_heads):
            x_h = x_heads[:, :, i, :]  # (B, T, head_dim)
            scores = head_attn(x_h)    # (B, T, 1)
            weights = F.softmax(scores, dim=1)
            context = torch.sum(x_h * weights, dim=1)  # (B, head_dim)
            contexts.append(context)

        output = torch.cat(contexts, dim=1)  # (B, feature_dim)
        return output


class SuspiciousActivityClassifier(nn.Module):
    """
    Multi-Scale 1D-CNN + Bidirectional LSTM + Multi-Head Temporal Attention.

    1. Multi-Scale Conv1D: Two conv layers with k=3 and k=5 for local and mid-range
       kinematic patterns, with gradual channel expansion (input→64→hidden).
    2. SE Channel Attention: Dynamically suppresses irrelevant feature channels.
    3. Bidirectional LSTM: Stronger gating (forget gate) captures onset-peak-decay
       pattern of violent events better than GRU.
    4. Multi-Head Attention (2 heads) + Max Pooling: Different heads attend to
       different phases of the action sequence.
    5. Dense MLP with LayerNorm and Dropout.
    """
    def __init__(
        self,
        input_size  = getattr(config, 'TIER2_INPUT_SIZE', 34),
        hidden_size = 64,  # Reduced from 128 to force generalization
        num_layers  = 1,   # Reduced from 2 to prevent overfitting
        dropout     = 0.3,
    ):
        super().__init__()

        mid_channels = 64  # Gradual expansion: input_size → 64 → hidden_size

        # 1. Multi-Scale 1D Temporal Convolution
        self.conv_block = nn.Sequential(
            # First conv: local patterns (kernel=3)
            nn.Conv1d(in_channels=input_size, out_channels=mid_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            # Second conv: mid-range patterns (kernel=5)
            nn.Conv1d(in_channels=mid_channels, out_channels=hidden_size,
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        # 2. Squeeze-and-Excitation channel attention
        self.se_block = SEBlock(hidden_size, reduction=4)

        # 3. Bidirectional LSTM (replaces GRU for stronger gating)
        self.lstm = nn.LSTM(
            input_size   = hidden_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # 4. Multi-Head Temporal Attention (2 heads)
        self.attention = MultiHeadTemporalAttention(hidden_size * 2, num_heads=2)

        # 5. Dense Classification Head (Lightweight)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),  # attention + max-pool concatenated
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Conv1d expects (batch, channels, seq_len)
        x_perm = x.permute(0, 2, 1)
        conv_out = self.conv_block(x_perm)       # (batch, hidden_size, seq_len)
        conv_out = self.se_block(conv_out)        # (batch, hidden_size, seq_len)
        conv_out = conv_out.permute(0, 2, 1)     # (batch, seq_len, hidden_size)

        lstm_out, _ = self.lstm(conv_out)         # (batch, seq_len, hidden_size * 2)

        # Combine attention context + max-pooled temporal representation
        att_context = self.attention(lstm_out)           # (batch, hidden_size * 2)
        max_context, _ = torch.max(lstm_out, dim=1)      # (batch, hidden_size * 2)
        combined = torch.cat([att_context, max_context], dim=1)  # (batch, hidden_size * 4)

        return self.classifier(combined)


# Keep backward-compatible alias
SuspiciousActivityGRU = SuspiciousActivityClassifier


class VisionBiLSTMClassifier(nn.Module):
    """
    MobileNetV2 Visual Features (1280-D) + Temporal Motion Delta + BiLSTM + Multi-Head Temporal Attention.
    Translates and enhances the architecture from cctv-classification.ipynb to PyTorch with dynamic motion flux.
    
    Architecture:
      1. Dynamic Temporal Motion Delta: delta_t = x_t - x_{t-1}, delta_0 = 0
      2. Dual Projection:
         - Spatial Semantics: 1280 -> 128 (LayerNorm, Linear, GELU, Dropout)
         - Motion Flux Delta: 1280 -> 128 (LayerNorm, Linear, GELU, Dropout)
         - Concatenation: 128 + 128 = 256
      3. Bidirectional LSTM: 2 layers (hidden_size=128, bidirectional -> 256)
      4. Multi-Head Temporal Attention (2 heads over 256-D) + Max-pooling context aggregation -> 512-D
      5. Deep Regularized MLP Head: 512 -> 128 -> 32 -> 1 with LayerNorm & Dropout
    """
    def __init__(
        self,
        embedding_dim: int = getattr(config, 'VISION_EMBEDDING_DIM', 1280),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.motion_norm = nn.LayerNorm(embedding_dim)

        # 1. Spatial feature projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 2. Motion velocity flux projection
        self.motion_proj = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 3. Bidirectional LSTM (input_size = 128 + 128 = 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 4. Multi-Head Temporal Attention (2 heads over 256-D)
        self.attention = MultiHeadTemporalAttention(hidden_size * 2, num_heads=2)

        # 5. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),  # 256 (att) + 256 (max) = 512
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len=15, 1280)
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]

        s_proj = self.spatial_proj(self.input_norm(x))
        m_proj = self.motion_proj(self.motion_norm(delta))
        fused = torch.cat([s_proj, m_proj], dim=-1)

        lstm_out, _ = self.lstm(fused)
        att_context = self.attention(lstm_out)
        max_context, _ = torch.max(lstm_out, dim=1)
        combined = torch.cat([att_context, max_context], dim=1)

        return self.classifier(combined)


class HybridClassifier(nn.Module):
    """
    Dual-Stream Vision + Kinematic Pose Fusion Classifier.
    Fuses visual semantics (MobileNetV2 1280-D) with geometric body dynamics (34-D keypoint motion).
    """
    def __init__(
        self,
        vision_dim: int = 1280,
        pose_dim: int = 34,
        vision_hidden: int = 128,
        pose_hidden: int = 64,
        dropout: float = 0.35,
    ):
        super().__init__()
        # Vision stream
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        self.vision_lstm = nn.LSTM(
            input_size=256,
            hidden_size=vision_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.vision_attn = MultiHeadTemporalAttention(vision_hidden * 2, num_heads=2)

        # Pose stream
        self.pose_conv = nn.Sequential(
            nn.Conv1d(pose_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.pose_lstm = nn.LSTM(
            input_size=64,
            hidden_size=pose_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.pose_attn = MultiHeadTemporalAttention(pose_hidden * 2, num_heads=2)

        # Gated Cross-Modal Fusion
        combined_dim = (vision_hidden * 2) + (pose_hidden * 2)  # 256 + 128 = 384
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, 1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_vision, x_pose):
        # Vision stream
        v_norm = self.vision_norm(x_vision)
        v_proj = self.vision_proj(v_norm)
        v_out, _ = self.vision_lstm(v_proj)
        v_ctx = self.vision_attn(v_out)

        # Pose stream
        p_perm = x_pose.permute(0, 2, 1)
        p_conv = self.pose_conv(p_perm).permute(0, 2, 1)
        p_out, _ = self.pose_lstm(p_conv)
        p_ctx = self.pose_attn(p_out)

        # Gated fusion
        combined = torch.cat([v_ctx, p_ctx], dim=1)
        g = self.gate(combined)
        fused = torch.cat([v_ctx * g, p_ctx * (1.0 - g)], dim=1)

        return self.classifier(fused)


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for hard example mining.
    Alpha=0.50 for balanced datasets (Fight=NonFight).
    """
    def __init__(self, alpha=0.50, gamma=2.0, label_smoothing=0.03):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)

        bce = - smoothed_targets * torch.log(inputs) - (1.0 - smoothed_targets) * torch.log(1.0 - inputs)
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        return torch.mean(alpha_t * focal_weight * bce)


def train_tier2(X_train, y_train, X_val, y_val,
                epochs=120, batch_size=64, device_str='cuda'):
    """
    Trains the Multi-Scale Conv-BiLSTM-Attention classifier with:
    - Feature scaling + heavy augmentation (jitter, dropout, temporal warp, mixup)
    - OneCycleLR scheduler
    - Stochastic Weight Averaging (SWA) for last 20% of epochs
    - Early stopping with patience=30
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"Training on Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Fit Feature Scaler on Training Set
    scaler = FeatureScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    scaler.save(SCALER_PATH)
    print(f"FeatureScaler fitted on {len(X_train)} samples and saved to {SCALER_PATH}")

    # 2. Build Model & Optimizers
    model = SuspiciousActivityClassifier().to(device)
    criterion = BinaryFocalLoss(alpha=0.50, gamma=2.0, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # OneCycleLR: proven single-cycle schedule for small datasets
    total_steps = epochs * (len(X_train) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        div_factor=10,          # start_lr = max_lr / 10 = 1e-4
        final_div_factor=1000,  # end_lr = start_lr / 1000 = 1e-7
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Stochastic Weight Averaging (last 20% of epochs)
    swa_start_epoch = int(epochs * 0.8)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=5)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = epochs  # Disable early stopping to allow OneCycleLR to finish
    patience_counter = 0
    use_swa = False

    os.makedirs(os.path.dirname(config.TIER2_MODEL_PATH), exist_ok=True)
    print(f"\nStarting Training | Train: {len(X_train)} | Val: {len(X_val)} "
          f"| Max Epochs: {epochs} | Batch: {batch_size} | SWA starts at epoch {swa_start_epoch}\n")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        if epoch >= swa_start_epoch:
            use_swa = True

        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            # Apply augmentation: standard transforms
            X_b_aug = TemporalAugmenter.augment(X_b, jitter_std=0.05, dropout_prob=0.15)

            # Apply mixup with 15% probability
            if not use_swa and torch.rand(1).item() < 0.15:
                X_b_aug, y_b = TemporalAugmenter.mixup(X_b_aug, y_b, alpha=0.4)

            optimizer.zero_grad()
            preds = model(X_b_aug)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Use appropriate scheduler
            if not use_swa:
                scheduler.step()

            train_loss += loss.item()
            predicted = (preds >= 0.5).float()
            train_correct += (predicted == (y_b >= 0.5).float()).sum().item()
            total_train += len(y_b)

        # Update SWA model
        if use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        train_loss /= len(train_loader)
        train_acc = (train_correct / total_train) * 100.0

        # Validate
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                val_preds = model(X_b)
                val_loss += criterion(val_preds, y_b).item()

                predicted = (val_preds >= 0.5).float()
                val_correct += (predicted == y_b).sum().item()
                total_val += len(y_b)

        val_loss /= len(val_loader)
        val_acc = (val_correct / total_val) * 100.0

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            swa_tag = " [SWA]" if use_swa else ""
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} ({train_acc:.1f}%) "
                  f"| Val: {val_loss:.4f} ({val_acc:.1f}%) | LR: {lr:.2e}{swa_tag}")

        # Checkpoint on best validation loss (only before SWA starts)
        if not use_swa and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config.TIER2_MODEL_PATH)
        elif not use_swa:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch+1}. "
                      f"Best Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.1f}%.")
                break

    # Finalize SWA: update batch norm and save
    if use_swa:
        print("\n[SWA] Updating batch normalization statistics...")
        update_bn(train_loader, swa_model, device=device)
        # Evaluate SWA model
        swa_model.eval()
        swa_val_loss, swa_val_correct, swa_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                swa_preds = swa_model(X_b)
                swa_val_loss += criterion(swa_preds, y_b).item()
                predicted = (swa_preds >= 0.5).float()
                swa_val_correct += (predicted == y_b).sum().item()
                swa_total += len(y_b)
        swa_val_acc = (swa_val_correct / swa_total) * 100.0
        swa_val_loss /= len(val_loader)
        print(f"[SWA] Val Loss: {swa_val_loss:.4f} (Acc: {swa_val_acc:.1f}%)")

        # Save SWA model if it improved
        if swa_val_loss < best_val_loss:
            best_val_loss = swa_val_loss
            best_val_acc = swa_val_acc
            # Save the inner model weights (not the SWA wrapper)
            torch.save(swa_model.module.state_dict(), config.TIER2_MODEL_PATH)
            print(f"[SWA] Improved! Saved SWA model.")
        else:
            print(f"[SWA] No improvement over best checkpoint (Val Loss: {best_val_loss:.4f}).")

    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f} "
          f"(Val Acc: {best_val_acc:.2f}%) -> {config.TIER2_MODEL_PATH}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(config.TIER2_MODEL_PATH, map_location=device, weights_only=False))
    return model


def train_vision_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cuda",
    checkpoint_path: str = getattr(config, 'VISION_MODEL_PATH', 'models/vision_bilstm.pt'),
    alpha: float = 0.50,
    target_recall: Optional[float] = None
) -> nn.Module:
    """
    Trains the VisionBiLSTMClassifier (MobileNetV2 + BiLSTM + Attention).
    
    Features:
      - Binary Focal Loss with customizable alpha (e.g. 0.65 for high fight recall)
      - AdamW Optimizer with Cosine Annealing LR
      - Gradient clipping
      - Target-recall or optimal Macro F1 threshold calibration
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"\n[Vision Classifier Training] Device: {device} " +
          (f"({torch.cuda.get_device_name(0)}) | FP16 AMP & TF32 Enabled | Alpha={alpha:.2f}" if device.type == 'cuda' else 'CPU'))

    model = VisionBiLSTMClassifier().to(device)
    criterion = BinaryFocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=use_amp, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        pin_memory=use_amp
    )

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_thresh = 0.50

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print(f"[Vision Classifier] Training: {len(X_train)} samples | Val: {len(X_val)} samples | Epochs: {epochs} | Batch: {batch_size}\n")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            # Feature jitter augmentation on embeddings
            if torch.rand(1).item() < 0.5:
                noise = torch.randn_like(X_b) * 0.02
                X_b = X_b + noise

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                preds = model(X_b)
                loss = criterion(preds, y_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            predicted = (preds >= 0.5).float()
            train_correct += (predicted == y_b).sum().item()
            total_train += len(y_b)

        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = (train_correct / total_train) * 100.0

        # Validate
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    preds = model(X_b)
                    loss_v = criterion(preds, y_b)
                val_loss += loss_v.item()

                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(y_b.cpu().numpy())

                predicted = (preds >= 0.5).float()
                val_correct += (predicted == y_b).sum().item()
                total_val += len(y_b)

        val_loss /= len(val_loader)
        val_acc = (val_correct / total_val) * 100.0
        val_preds_arr = np.vstack(all_val_preds).flatten()
        val_targets_arr = np.vstack(all_val_targets).flatten()

        # Find best threshold on validation predictions
        cur_best_f1 = 0.0
        cur_best_thresh = 0.50
        for t in np.linspace(0.20, 0.70, 26):
            p = (val_preds_arr >= t).astype(int)
            tp = np.sum((p == 1) & (val_targets_arr == 1))
            fp = np.sum((p == 1) & (val_targets_arr == 0))
            fn = np.sum((p == 0) & (val_targets_arr == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            if target_recall is not None:
                if rec >= target_recall and f1 > cur_best_f1:
                    cur_best_f1 = f1
                    cur_best_thresh = t
            else:
                if f1 > cur_best_f1:
                    cur_best_f1 = f1
                    cur_best_thresh = t

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1 or epoch == 0:
            lr_curr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} ({train_acc:.1f}%) "
                  f"| Val Loss: {val_loss:.4f} ({val_acc:.1f}%) | Val F1: {cur_best_f1*100:.1f}% | LR: {lr_curr:.2e}")

        # Checkpoint on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = cur_best_f1
            best_thresh = cur_best_thresh
            torch.save({
                'model_state_dict': model.state_dict(),
                'calibrated_threshold': float(best_thresh),
                'val_f1': float(best_val_f1),
                'val_loss': float(best_val_loss),
            }, checkpoint_path)

    print(f"\n[Vision Classifier] Training Complete. Best Val Loss: {best_val_loss:.4f} "
          f"(F1: {best_val_f1*100:.2f}%, Thresh: {best_thresh:.2f}) -> {checkpoint_path}")

    # Load best weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.calibrated_threshold = float(ckpt.get('calibrated_threshold', 0.50))
    return model


class UnifiedInferencer:
    """
    Production-Ready Real-Time Inference Engine for Sentinel AI.
    
    Supports:
      1. Vision Stream (`push_frame`): Takes raw camera frames, runs MobileNetV2 per frame
         in a rolling 15-frame buffer, and predicts via VisionBiLSTMClassifier.
      2. Kinematic Stream (`push_features`): Takes 34-D pose vectors (Tier-2 backward compatible).
      3. Hybrid Stream (`push_multimodal`): Dual-stream fusion.
    """
    def __init__(self, model_type: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type or getattr(config, 'MODEL_TYPE', 'vision_bilstm')
        
        # Vision Stream Components
        self.backbone = None
        self.pool = None
        self.vision_model = None
        self.frame_buffer = []
        self.roi_buffer = []
        self.vision_frame_count = getattr(config, 'VISION_FRAME_COUNT', 15)
        self.vision_img_size = getattr(config, 'VISION_IMG_SIZE', 128)
        self.buffer_horizon = getattr(config, 'INFERENCE_BUFFER_FRAMES', 30)
        self.eval_interval = getattr(config, 'INFERENCE_EVAL_INTERVAL', 2)
        self.enable_roi = getattr(config, 'ENABLE_ROI_INSPECTION', True)
        self.frame_counter = 0
        self.latest_prob = 0.0
        self.calibrated_threshold = 0.50

        # Pose Stream Components
        self.pose_model = None
        self.scaler = FeatureScaler()
        self.feature_buffer = []
        self.pose_window_size = getattr(config, 'FEATURE_WINDOW_FRAMES', 30)

        self._init_models()

    def _init_models(self):
        # 1. Initialize Vision Model if needed
        if self.model_type in ("vision_bilstm", "hybrid"):
            try:
                from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                weights = MobileNet_V2_Weights.DEFAULT
                mobilenet = mobilenet_v2(weights=weights)
                self.backbone = mobilenet.features.to(self.device).eval()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

                self.vision_model = VisionBiLSTMClassifier().to(self.device)
                v_path = getattr(config, 'VISION_MODEL_PATH', 'models/vision_bilstm.pt')
                if os.path.exists(v_path):
                    ckpt = torch.load(v_path, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        self.vision_model.load_state_dict(ckpt['model_state_dict'])
                        self.calibrated_threshold = float(ckpt.get('calibrated_threshold', 0.50))
                    else:
                        self.vision_model.load_state_dict(ckpt)
                    print(f"[Inferencer] Loaded Vision-BiLSTM weights from {v_path} (Threshold: {self.calibrated_threshold:.2f})")
                else:
                    print(f"[Inferencer] Notice: Vision checkpoint not found at {v_path}, run training first.")
                self.vision_model.eval()
            except Exception as e:
                print(f"[Inferencer] Warning: could not load vision model ({e})")

        # 2. Initialize Pose Model if needed
        if self.model_type in ("pose_gru", "hybrid") or self.vision_model is None:
            if os.path.exists(SCALER_PATH):
                self.scaler.load(SCALER_PATH)
            self.pose_model = SuspiciousActivityClassifier().to(self.device)
            p_path = getattr(config, 'TIER2_MODEL_PATH', 'models/tier2_gru.pt')
            if os.path.exists(p_path):
                self.pose_model.load_state_dict(torch.load(p_path, map_location=self.device, weights_only=False))
                print(f"[Inferencer] Loaded Pose-BiLSTM weights from {p_path}")
            self.pose_model.eval()

    def push_frame(self, frame: np.ndarray, persons: Optional[List[Dict]] = None) -> Optional[float]:
        """
        Pushes a single BGR camera frame into the rolling multi-scale temporal window.
        
        Features:
          1. Uniform strided sampling across the rolling horizon (1.0 - 1.5s real action).
          2. Dual-scale evaluation: evaluates full camera scene AND zoomed-in person cluster (ROI).
          3. Production Gating:
             - 0 people: Instant confidence = 0.0 (fist fights require humans).
             - 1 person: Confidence scaled down < 0.15 (cannot brawl with oneself).
             - Scene motion difference check prevents static textures from triggering alerts.
        """
        if self.backbone is None or self.vision_model is None:
            return None

        # Production Gate 1: Zero persons detected -> strict 0.0%
        if persons is not None and len(persons) == 0:
            self.latest_prob = 0.0
            return 0.0

        self.frame_counter += 1

        # 1. Preprocess full scene
        small = cv2.resize(frame, (self.vision_img_size, self.vision_img_size))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        self.frame_buffer.append(chw)
        if len(self.frame_buffer) > self.buffer_horizon:
            self.frame_buffer.pop(0)

        # 2. Preprocess Person ROI Cluster
        if self.enable_roi and persons and len(persons) > 0:
            h_img, w_img = frame.shape[:2]
            bx1 = max(0, min(p['bbox'][0] for p in persons))
            by1 = max(0, min(p['bbox'][1] for p in persons))
            bx2 = min(w_img, max(p['bbox'][2] for p in persons))
            by2 = min(h_img, max(p['bbox'][3] for p in persons))

            # Add 20% margin
            bw = bx2 - bx1
            bh = by2 - by1
            mx = int(bw * 0.20)
            my = int(bh * 0.20)
            rx1 = max(0, bx1 - mx)
            ry1 = max(0, by1 - my)
            rx2 = min(w_img, bx2 + mx)
            ry2 = min(h_img, by2 + my)

            if (rx2 - rx1) > 40 and (ry2 - ry1) > 40:
                crop = frame[ry1:ry2, rx1:rx2]
                small_roi = cv2.resize(crop, (self.vision_img_size, self.vision_img_size))
                rgb_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                chw_roi = np.transpose(rgb_roi, (2, 0, 1))
                self.roi_buffer.append(chw_roi)
            else:
                self.roi_buffer.append(chw)
        else:
            self.roi_buffer.append(chw)

        if len(self.roi_buffer) > self.buffer_horizon:
            self.roi_buffer.pop(0)

        # 3. Evaluate when buffer is populated
        if len(self.frame_buffer) >= self.vision_frame_count:
            if self.frame_counter % self.eval_interval == 0:
                indices = np.linspace(0, len(self.frame_buffer) - 1, self.vision_frame_count, dtype=int)
                win_scene = np.array([self.frame_buffer[i] for i in indices], dtype=np.float32)

                # Batch scene + ROI if distinct
                has_roi = (self.enable_roi and len(self.roi_buffer) == len(self.frame_buffer) and persons and len(persons) >= 1)
                if has_roi:
                    win_roi = np.array([self.roi_buffer[i] for i in indices], dtype=np.float32)
                    batch = np.stack([win_scene, win_roi], axis=0)
                else:
                    batch = np.expand_dims(win_scene, axis=0)

                tensor_batch = torch.FloatTensor(batch).to(self.device)
                with torch.no_grad():
                    B, T, C, H, W = tensor_batch.shape
                    flat = tensor_batch.view(B * T, C, H, W)
                    feats = self.pool(self.backbone(flat)).flatten(1).view(B, T, -1)
                    probs = self.vision_model(feats).squeeze(1).cpu().numpy()
                    raw_prob = float(np.max(probs)) if np.ndim(probs) > 0 else float(probs)

                    # Production Gate 2: Single person present -> scale down (cannot brawl alone)
                    if persons is not None and len(persons) == 1:
                        raw_prob = min(raw_prob * 0.15, 0.10)

                    self.latest_prob = raw_prob

            return self.latest_prob

        return None

    def push_features(self, feature_vector: np.ndarray, person_count: int = 0) -> Optional[float]:
        """
        Pushes a 34-D kinematic feature vector (Tier-2 backward compatible).
        Gated by person count to eliminate false alarms when 0 or 1 person is present.
        """
        if self.pose_model is None:
            return None

        # Production Gate: No fight possible with 0 people
        if person_count == 0:
            return 0.0
        if person_count == 1:
            return 0.05

        self.feature_buffer.append(feature_vector)
        if len(self.feature_buffer) >= self.pose_window_size:
            window = np.array(self.feature_buffer[-self.pose_window_size:], dtype=np.float32)
            window_norm = self.scaler.transform(window)
            x = torch.FloatTensor(window_norm).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = self.pose_model(x).item()
            return float(prob)

        return None

    def reset(self):
        """Clears rolling buffers."""
        self.frame_buffer.clear()
        self.roi_buffer.clear()
        self.feature_buffer.clear()
        self.frame_counter = 0
        self.latest_prob = 0.0


# Backward-compatible alias
Tier2Inferencer = UnifiedInferencer
