"""
Step 2 training.

Loads the pre-extracted feature arrays (output of extract_features.py),
splits them into train/val/test sets (stratified so each split keeps
a proportional mix of both datasets), and trains the Tier 2 GRU classifier.

Run from the project root:
    .\\venv\\Scripts\\python scripts/train.py

Expects:
    data/X_tier2.npy      (produced by extract_features.py)
    data/y_tier2.npy
    data/sources.npy      (dataset name per clip -> optional but used for reporting)

Saves:
    models/tier2_gru.pt   (best checkpoint by validation loss)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pathlib import Path

import config
from src.classifier import train_tier2


def stratified_split(X, y, sources, train_ratio=0.80, val_ratio=0.10, seed=42):
    """
    Split arrays into train/val/test while keeping the same class ratio in each split.

    Why stratified? If we just shuffle and cut, we might end up with 90% Fight in
    the test set by bad luck. Stratification guarantees each split mirrors the
    original class distribution.

    Args:
        X, y, sources : arrays of the same length N
        train_ratio   : fraction for training (default 80%)
        val_ratio     : fraction for validation (default 10%)
                        remaining (10%) goes to test

    Returns:
        six arrays: X_train, y_train, X_val, y_val, X_test, y_test
        plus sources_test for cross-dataset analysis
    """
    rng = np.random.default_rng(seed=seed)

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []
    src_test_list              = []

    # Split independently for each class (0 and 1) to preserve proportions
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)               # shuffle within class

        n        = len(idx)
        n_train  = int(n * train_ratio)
        n_val    = int(n * val_ratio)

        X_train_list.append(X[idx[:n_train]])
        y_train_list.append(y[idx[:n_train]])

        X_val_list.append(X[idx[n_train:n_train + n_val]])
        y_val_list.append(y[idx[n_train:n_train + n_val]])

        X_test_list.append(X[idx[n_train + n_val:]])
        y_test_list.append(y[idx[n_train + n_val:]])
        src_test_list.append(sources[idx[n_train + n_val:]])

    # Concatenate and shuffle the final train set
    # (otherwise it's [all fight, all nonfight] which biases the GRU training)
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    train_idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    X_val     = np.concatenate(X_val_list)
    y_val     = np.concatenate(y_val_list)

    X_test    = np.concatenate(X_test_list)
    y_test    = np.concatenate(y_test_list)
    src_test  = np.concatenate(src_test_list)

    return X_train, y_train, X_val, y_val, X_test, y_test, src_test


def main():
    # Load pre-extracted features 
    x_path   = Path("data/X_tier2.npy")
    y_path   = Path("data/y_tier2.npy")
    src_path = Path("data/sources.npy")

    if not x_path.exists() or not y_path.exists():
        print("ERROR: Feature arrays not found.")
        print("  Run first:  .\\venv\\Scripts\\python scripts/extract_features.py")
        return

    X       = np.load(x_path)
    y       = np.load(y_path)
    sources = np.load(src_path, allow_pickle=True) if src_path.exists() else np.array(["unknown"] * len(X))

    print(f"Loaded arrays:")
    print(f"  X       : {X.shape}")
    print(f"  y       : {y.shape}   Fight={int(y.sum())}  NonFight={int((y==0).sum())}")

    # Per-dataset breakdown
    for src in sorted(set(sources)):
        mask = sources == src
        print(f"  {src:<14}  Fight: {int(y[mask].sum()):>5}   NonFight: {int((y[mask]==0).sum()):>5}")
    print()

    if X.shape[2] != config.TIER2_INPUT_SIZE:
        print(f"ERROR: Feature size mismatch!")
        print(f"  X has {X.shape[2]} features per frame but config.TIER2_INPUT_SIZE = {config.TIER2_INPUT_SIZE}")
        print(f"  Fix: set TIER2_INPUT_SIZE = {X.shape[2]} in config.py, then rerun.")
        return

    # Stratified split 
    X_train, y_train, X_val, y_val, X_test, y_test, src_test = stratified_split(
        X, y, sources, train_ratio=0.80, val_ratio=0.10
    )
    print(f"Split (stratified 80/10/10):")
    print(f"  train : {len(X_train):>5}  (Fight={int(y_train.sum())}, NonFight={int((y_train==0).sum())})")
    print(f"  val   : {len(X_val):>5}  (Fight={int(y_val.sum())}, NonFight={int((y_val==0).sum())})")
    print(f"  test  : {len(X_test):>5}  (Fight={int(y_test.sum())}, NonFight={int((y_test==0).sum())})")
    print()

    # Training
    print("Starting training with Conv-BiGRU-Attention + Feature Scaling + Data Augmentation...\n")
    model = train_tier2(X_train, y_train, X_val, y_val, epochs=80)

    # Test-set evaluation with normalized features
    import torch
    from src.classifier import FeatureScaler, SCALER_PATH
    scaler = FeatureScaler()
    scaler.load(SCALER_PATH)
    
    X_val_norm  = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    device = next(model.parameters()).device
    model.eval()

    # 1. Calibrate threshold on validation set
    with torch.no_grad():
        val_probs = model(torch.FloatTensor(X_val_norm).to(device)).squeeze(1).cpu().numpy()
        test_probs = model(torch.FloatTensor(X_test_norm).to(device)).squeeze(1).cpu().numpy()

    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.35, 0.65, 31):
        v_pred = (val_probs >= t).astype(int)
        tp = np.sum((v_pred == 1) & (y_val == 1))
        fp = np.sum((v_pred == 1) & (y_val == 0))
        fn = np.sum((v_pred == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    # 2. Evaluate on held-out test set
    preds = (test_probs >= best_thresh).astype(int)
    correct = (preds == y_test.astype(int)).sum()
    overall_acc = correct / len(y_test) * 100

    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))

    prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    rec  = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1_s = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  HELD-OUT TEST SET EVALUATION")
    print(f"{'='*55}")
    print(f"  Calibrated Threshold : {best_thresh:.2f} (Val F1: {best_f1*100:.1f}%)")
    print(f"  Overall Accuracy     : {overall_acc:.2f}%  ({correct}/{len(y_test)})")
    print(f"  Precision            : {prec:.2f}%")
    print(f"  Recall (Sensitivity) : {rec:.2f}%")
    print(f"  F1-Score             : {f1_s:.2f}%")
    print(f"  False Positive Rate  : {(fp/(fp+tn)*100):.2f}%")
    print(f"  Confusion Matrix     : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"{'-'*55}")

    # Per-dataset accuracy
    for src in sorted(set(src_test)):
        mask  = src_test == src
        n     = mask.sum()
        acc   = (preds[mask] == y_test[mask].astype(int)).sum() / n * 100
        print(f"  {src:<14}  acc: {acc:.2f}%  ({n} clips)")

    print(f"\n  Model saved to: {config.TIER2_MODEL_PATH}")
    print(f"  Feature Scaler: {SCALER_PATH}")
    print(f"  Run the pipeline:  python app.py\n")


if __name__ == "__main__":
    main()

