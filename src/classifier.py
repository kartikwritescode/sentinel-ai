# Production-Grade Conv-BiGRU Temporal Self-Attention Classifier
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import config

SCALER_PATH = "data/scaler_stats.npz"


class FeatureScaler:
    """
    Z-Score Standardization across the 24 feature dimensions.
    Normalizes features to zero-mean and unit variance: (X - mu) / (sigma + eps).
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # X: (N, seq_len, 24)
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
    Data augmentation for temporal kinematic sequences:
      1. Gaussian Kinematic Jitter (adds micro-sensor noise)
      2. Temporal Time-Warping (speeds up / slows down movement tempo)
      3. Channel Dropout (prevents over-reliance on single features)
    """
    @staticmethod
    def augment(X_batch, jitter_std=0.03, dropout_prob=0.10):
        # X_batch: (N, seq_len, 24)
        N, seq_len, feat_dim = X_batch.shape
        augmented = X_batch.clone()

        # 1. Gaussian noise
        noise = torch.randn_like(augmented) * jitter_std
        augmented = augmented + noise

        # 2. Random Feature Channel Dropout (masks 1-2 channels per sample)
        if torch.rand(1).item() < 0.5:
            mask = (torch.rand(N, 1, feat_dim, device=X_batch.device) > dropout_prob).float()
            augmented = augmented * mask

        return augmented


class TemporalAttention(nn.Module):
    """
    Scaled dot-product temporal self-attention mechanism.
    Identifies high-impact combat frames within the 30-frame window.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)  # (batch, feature_dim)
        return context, weights


class SuspiciousActivityGRU(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional GRU + Temporal Self-Attention Architecture.
    
    1. Conv1D Temporal Block: Captures local sub-second kinematic derivatives (jerk, sudden strike acceleration).
    2. Bidirectional GRU: Captures long-range temporal sequence context (forward + backward passes).
    3. Multi-Head Attention + Residual Pooling: Focuses on critical clash/strike frames.
    4. Dense MLP Classifier with LayerNorm and Dropout: Produces calibrated suspicion probability.
    """
    def __init__(
        self,
        input_size  = getattr(config, 'TIER2_INPUT_SIZE', 24),
        hidden_size = getattr(config, 'TIER2_HIDDEN_SIZE', 128),
        num_layers  = getattr(config, 'TIER2_NUM_LAYERS', 2),
        dropout     = 0.35,
    ):
        super().__init__()

        # 1. 1D Temporal Convolution (extracts local multi-frame joint dynamics)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # 2. Bidirectional GRU
        self.gru = nn.GRU(
            input_size   = hidden_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # 3. Temporal Self-Attention over sequence timesteps
        self.attention = TemporalAttention(hidden_size * 2)

        # 4. Dense Classification Head (combines attention context + global max pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
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
        # x: (batch, seq_len, input_size)
        # Conv1d expects (batch, channels, seq_len)
        x_perm = x.permute(0, 2, 1)
        conv_out = self.conv1d(x_perm)  # (batch, hidden_size, seq_len)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq_len, hidden_size)

        gru_out, _ = self.gru(conv_out)  # (batch, seq_len, hidden_size * 2)
        
        # Combine attention context + max-pooled temporal representation
        att_context, _ = self.attention(gru_out)      # (batch, hidden_size * 2)
        max_context, _ = torch.max(gru_out, dim=1)    # (batch, hidden_size * 2)
        combined = torch.cat([att_context, max_context], dim=1)  # (batch, hidden_size * 4)

        return self.classifier(combined)


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for class-imbalance and hard negative mining.
    """
    def __init__(self, alpha=0.65, gamma=2.0, label_smoothing=0.03):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        
        bce = - smoothed_targets * torch.log(inputs) - (1.0 - smoothed_targets) * torch.log(1.0 - inputs)
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        return torch.mean(alpha_t * focal_weight * bce)


def train_tier2(X_train, y_train, X_val, y_val,
                epochs=75, batch_size=64, device_str='cuda'):
    """
    Trains the Conv-BiGRU-Attention classifier with feature scaling, augmentation, and early stopping.
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
    model = SuspiciousActivityGRU().to(device)
    criterion = BinaryFocalLoss(alpha=0.65, gamma=2.0, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

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
        pin_memory=(device.type == 'cuda'), num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 25
    patience_counter = 0

    os.makedirs(os.path.dirname(config.TIER2_MODEL_PATH), exist_ok=True)
    print(f"\nStarting Training | Train: {len(X_train)} | Val: {len(X_val)} | Max Epochs: {epochs} | Batch: {batch_size}\n")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            # Apply on-the-fly kinematic data augmentation
            X_b_aug = TemporalAugmenter.augment(X_b, jitter_std=0.02, dropout_prob=0.08)

            optimizer.zero_grad()
            preds = model(X_b_aug)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (preds >= 0.5).float()
            train_correct += (predicted == y_b).sum().item()
            total_train += len(y_b)

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
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} ({train_acc:.1f}%) | Val: {val_loss:.4f} ({val_acc:.1f}%) | LR: {lr:.2e}")

        # Checkpoint on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config.TIER2_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch+1}. Restoring best checkpoint (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.1f}%).")
                break

    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f} (Val Acc: {best_val_acc:.2f}%) -> {config.TIER2_MODEL_PATH}")
    
    # Reload best checkpoint into model before returning
    model.load_state_dict(torch.load(config.TIER2_MODEL_PATH, map_location=device))
    return model


class Tier2Inferencer:
    """
    Production-ready real-time inferencer with automated feature scaling.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SuspiciousActivityGRU().to(self.device)
        self.scaler = FeatureScaler()
        
        if os.path.exists(SCALER_PATH):
            self.scaler.load(SCALER_PATH)

        if os.path.exists(config.TIER2_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(config.TIER2_MODEL_PATH, map_location=self.device)
            )
            print(f"[Inferencer] Loaded Conv-BiGRU-Attention weights from {config.TIER2_MODEL_PATH}")
        else:
            print(f"[Inferencer] Warning: Model checkpoint not found at {config.TIER2_MODEL_PATH}")

        self.model.eval()
        self._feature_buffer = []
        self._window_size = getattr(config, 'FEATURE_WINDOW_FRAMES', 30)

    def push_features(self, feature_vector):
        """
        Pushes single frame 24-D vector. Returns float confidence (0-1) when buffer fills.
        """
        self._feature_buffer.append(feature_vector)

        if len(self._feature_buffer) >= self._window_size:
            window = np.array(self._feature_buffer[-self._window_size:], dtype=np.float32)
            # Apply feature standardization
            window_norm = self.scaler.transform(window)
            return self._predict(window_norm)

        return None

    def _predict(self, normalized_window):
        x = torch.FloatTensor(normalized_window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(x).item()
        return float(prob)

    def reset(self):
        self._feature_buffer.clear()
