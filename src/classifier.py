# GRU model — GPU-accelerated with DataLoader mini-batch training

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import config


class SuspiciousActivityGRU(nn.Module):
    """
    Bidirectional GRU classifier for suspicious activity detection.

    Input shape:  (batch_size, sequence_length, input_size)
    Output shape: (batch_size, 1)  — probability of being suspicious (0–1)

    Bidirectional = reads the sequence forwards AND backwards, then
    concatenates both hidden states → full temporal context.
    """

    def __init__(
        self,
        input_size  = config.TIER2_INPUT_SIZE,
        hidden_size = config.TIER2_HIDDEN_SIZE,
        num_layers  = config.TIER2_NUM_LAYERS,
        dropout     = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,        # (batch, seq, features) — more intuitive
            bidirectional= True,        # forward + backward pass
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional doubles the output size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # output is a probability
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)
        # gru_out: (batch, seq_len, hidden_size * 2)
        # Take only the LAST timestep — it has seen the full sequence
        last_out = gru_out[:, -1, :]     # (batch, hidden_size * 2)
        return self.classifier(last_out)  # (batch, 1)


# Training (standalone, GPU-accelerated with DataLoader) 

def train_tier2(X_train, y_train, X_val, y_val,
                epochs=None, batch_size=None, device_str=None):
    """
    Train the GRU with mini-batch gradient descent on GPU (or CPU fallback).

    DataLoader batching is critical for GPU utilisation — full-batch training
    sends one giant tensor per step and the GPU sits mostly idle; mini-batches
    keep thousands of CUDA cores busy continuously.

    Args:
        X_train / y_train : np.ndarray — training data (N, seq_len, features)
        X_val   / y_val   : np.ndarray — validation data
        epochs            : int  — defaults to config.TRAINING_EPOCHS
        batch_size        : int  — defaults to config.TRAINING_BATCH_SIZE
        device_str        : str  — 'cuda' or 'cpu', defaults to config.TRAINING_DEVICE

    Returns:
        Trained SuspiciousActivityGRU (best checkpoint saved to config.TIER2_MODEL_PATH)
    """
    epochs     = epochs     or config.TRAINING_EPOCHS
    batch_size = batch_size or config.TRAINING_BATCH_SIZE
    device_str = device_str or config.TRAINING_DEVICE

    # Resolve device — fail loudly if CUDA requested but not available
    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested (config.TRAINING_DEVICE='cuda') but no GPU found.\n"
                "Either: install CUDA PyTorch, or set TRAINING_DEVICE='cpu' in config.py."
            )
        device = torch.device('cuda')
        print(f"Training on: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB VRAM)")
    else:
        device = torch.device('cpu')
        print("Training on: CPU")

    #  Model Building 
    model     = SuspiciousActivityGRU().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler: halve LR if val_loss stops improving for 10 epochs
    # This squeezes extra accuracy out without manually tuning the LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # DataLoaders (mini-batch) 
    # TensorDataset pairs each X sample with its y label
    # DataLoader shuffles + batches automatically, pin_memory speeds up GPU transfers
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1),   # (N,) → (N,1)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,                     # shuffle every epoch — prevents order bias
        pin_memory=(device.type == 'cuda'),  # pre-pin RAM for faster GPU transfer
        num_workers=0,                    # 0 on Windows (multiprocessing issues)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,        # val can use bigger batches (no gradients)
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=0,
    )

    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(config.TIER2_MODEL_PATH), exist_ok=True)

    print(f"\nBatch size: {batch_size}  |  Batches/epoch: {len(train_loader)}  |  Epochs: {epochs}\n")

    for epoch in range(epochs):

        # Train 
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)  # non_blocking with pin_memory
            y_batch = y_batch.to(device, non_blocking=True)  # = async GPU transfer

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)   # average over all batches

        # Validate 
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                val_preds = model(X_batch)
                val_loss += criterion(val_preds, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)   # adjust LR if plateau

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.TIER2_MODEL_PATH)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  ->  {config.TIER2_MODEL_PATH}")
    return model


# Inference 

class Tier2Inferencer:
    """
    Loads the trained GRU and runs real-time inference on a rolling
    window of feature vectors.  Separate from training so the live
    pipeline imports only inference code — no training deps.
    """

    def __init__(self):
        # Inference runs on CPU by default for live pipeline portability
        # (free deployment hosts have no GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = SuspiciousActivityGRU().to(self.device)
        self.model.load_state_dict(
            torch.load(config.TIER2_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()   # ALWAYS eval mode for inference — disables dropout

        self._feature_buffer = []
        self._window_size    = config.FEATURE_WINDOW_FRAMES

    def push_features(self, feature_vector):
        """
        Push one frame's feature vector. Returns a suspicion probability
        (float 0–1) once the buffer fills, otherwise None.
        """
        self._feature_buffer.append(feature_vector)

        if len(self._feature_buffer) >= self._window_size:
            window = self._feature_buffer[-self._window_size:]
            return self._predict(np.array(window))

        return None

    def _predict(self, feature_window):
        """
        Args:
            feature_window: np.ndarray  (window_size, input_size)
        Returns:
            float — probability of suspicious activity
        """
        # unsqueeze(0) adds the batch dim: (seq, feat) → (1, seq, feat)
        x = torch.FloatTensor(feature_window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(x).item()
