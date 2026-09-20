"""
scripts/train.py
────────────────
Production Training Pipeline for Sentinel AI.

Supports:
  1. Vision-BiLSTM Classifier (--model vision_bilstm):
     High-performing MobileNetV2 + BiLSTM + Attention architecture based on cctv-classification.ipynb.
  2. Kinematic Pose Classifier (--model pose_gru):
     Tier-2 Conv1D-BiLSTM on 34-D scale-normalized pose & dynamics features.
  3. Hybrid Dual-Stream Classifier (--model hybrid):
     Fuses vision and kinematic features via gated cross-modal attention.

Usage Examples:
  python scripts/train.py --model vision_bilstm --epochs 35 --batch-size 32
  python scripts/train.py --model vision_bilstm --dataset scvd --epochs 30
  python scripts/train.py --model pose_gru --epochs 120 --batch-size 64
"""

import sys
import os
import argparse
import json
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import config
from src.classifier import (
    VisionBiLSTMClassifier,
    SuspiciousActivityClassifier,
    FeatureScaler,
    SCALER_PATH,
    train_vision_classifier,
    train_tier2
)


def stratified_split(X, y, sources, clip_ids=None,
                     train_ratio=0.80, val_ratio=0.10, seed=42):
    """
    Split arrays into train/val/test while keeping:
    1. The same class ratio in each split (stratified)
    2. All crops from the same clip in the same split (clip-aware)
    """
    rng = np.random.default_rng(seed=seed)

    if clip_ids is None:
        clip_ids = np.arange(len(X))

    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    X_test_list,  y_test_list  = [], []
    src_test_list              = []
    cid_test_list              = []

    for cls in [0, 1]:
        cls_mask = (y == cls)
        cls_clip_ids = clip_ids[cls_mask]
        unique_clips = np.unique(cls_clip_ids)
        unique_clips = rng.permutation(unique_clips)

        n_clips    = len(unique_clips)
        n_train    = int(n_clips * train_ratio)
        n_val      = int(n_clips * val_ratio)

        train_clips = set(unique_clips[:n_train])
        val_clips   = set(unique_clips[n_train:n_train + n_val])
        test_clips  = set(unique_clips[n_train + n_val:])

        for idx in range(len(X)):
            if y[idx] != cls:
                continue
            cid = clip_ids[idx]
            if cid in train_clips:
                X_train_list.append(X[idx])
                y_train_list.append(y[idx])
            elif cid in val_clips:
                X_val_list.append(X[idx])
                y_val_list.append(y[idx])
            elif cid in test_clips:
                X_test_list.append(X[idx])
                y_test_list.append(y[idx])
                src_test_list.append(sources[idx])
                cid_test_list.append(cid)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    train_idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    X_val     = np.array(X_val_list)
    y_val     = np.array(y_val_list)
    X_test    = np.array(X_test_list)
    y_test    = np.array(y_test_list)
    src_test  = np.array(src_test_list)
    cid_test  = np.array(cid_test_list)

    return X_train, y_train, X_val, y_val, X_test, y_test, src_test, cid_test


def evaluate_and_log(model, X_test, y_test, src_test, cid_test, calibrated_thresh=0.50, model_name="vision_bilstm"):
    """Evaluates trained model on test set and saves eval report."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        test_probs = model(torch.FloatTensor(X_test).to(device)).squeeze(1).cpu().numpy()

    preds = (test_probs >= calibrated_thresh).astype(int)
    targets = y_test.astype(int)

    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    total = len(targets)
    acc = (tp + tn) / total * 100.0 if total > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    macro_f1 = f1_score(targets, preds, average="macro") * 100.0
    weighted_f1 = f1_score(targets, preds, average="weighted") * 100.0
    fpr = fp / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  HELD-OUT TEST SET EVALUATION ({model_name.upper()})")
    print(f"{'='*65}")
    print(f"  Calibrated Threshold : {calibrated_thresh:.2f}")
    print(f"  Test Set Size        : {total} samples")
    print(f"  -------------------------------------------------------------")
    print(f"  Accuracy             : {acc:.2f}%")
    print(f"  Macro F1             : {macro_f1:.2f}%")
    print(f"  Weighted F1          : {weighted_f1:.2f}%")
    print(f"  Precision            : {prec:.2f}%")
    print(f"  Recall (Sensitivity) : {rec:.2f}%")
    print(f"  Specificity          : {spec:.2f}%")
    print(f"  False Positive Rate  : {fpr:.2f}%")
    print(f"  -------------------------------------------------------------")
    print(f"  Confusion Matrix     : TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"  -------------------------------------------------------------")

    # Per-dataset breakdown
    dataset_metrics = {}
    print(f"  Per-Dataset Accuracy:")
    for src in sorted(set(src_test)):
        mask = (src_test == src)
        sub_correct = np.sum(preds[mask] == targets[mask])
        sub_total = np.sum(mask)
        sub_acc = sub_correct / sub_total * 100.0 if sub_total > 0 else 0.0
        sub_f1 = f1_score(targets[mask], preds[mask], average="macro", zero_division=0) * 100.0
        print(f"    {str(src):<14}: {sub_acc:6.2f}% (Acc) | {sub_f1:6.2f}% (Macro F1) | {sub_correct}/{sub_total} clips")
        dataset_metrics[str(src)] = {
            "total": int(sub_total),
            "correct": int(sub_correct),
            "accuracy_pct": round(sub_acc, 2),
            "macro_f1_pct": round(sub_f1, 2)
        }

    report = {
        "model_name": model_name,
        "calibrated_threshold": round(calibrated_thresh, 2),
        "test_samples": total,
        "metrics": {
            "accuracy_pct": round(acc, 2),
            "macro_f1_pct": round(macro_f1, 2),
            "weighted_f1_pct": round(weighted_f1, 2),
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "specificity_pct": round(spec, 2),
            "false_positive_rate_pct": round(fpr, 2)
        },
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        },
        "per_dataset": dataset_metrics
    }

    report_path = Path("data/eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n  Saved full evaluation report to: {report_path.resolve()}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI Model Training Entry Point")
    parser.add_argument("--model", type=str, default="vision_bilstm",
                        choices=["vision_bilstm", "pose_gru"],
                        help="Model architecture to train: 'vision_bilstm' or 'pose_gru'")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "scvd", "rlvs", "rwf"],
                        help="Dataset selection: 'all' (Unified RLVS+RWF+SCVD), 'scvd', 'rlvs', 'rwf'")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: 35 for vision, 120 for pose)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 32 for vision, 64 for pose)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--focal-alpha", type=float, default=0.50,
                        help="Focal Loss alpha for violence class (use 0.65 - 0.70 for max fight recall)")
    parser.add_argument("--target-recall", type=float, default=None,
                        help="Target validation recall (e.g. 0.98 to guarantee zero missed fights)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'='*65}")
    print(f"  SENTINEL AI -- MODEL TRAINING PIPELINE (GPU ACCELERATED)")
    print(f"{'='*65}")
    print(f"  Model Architecture : {args.model}")
    print(f"  Dataset Scope      : {args.dataset.upper()}")

    # 1. Load Data Arrays
    if args.model == "vision_bilstm":
        epochs = args.epochs if args.epochs is not None else 35
        batch_size = args.batch_size if args.batch_size is not None else 64
        x_path = Path("data/X_vision.npy")
        y_path = Path("data/y_vision.npy")
        src_path = Path("data/sources_vision.npy")
        cid_path = Path("data/clip_ids_vision.npy")

        if not x_path.exists():
            print(f"\nERROR: Vision features not found at {x_path}.")
            print("  Run first: python scripts/extract_vision_features.py\n")
            return
    else:
        epochs = args.epochs if args.epochs is not None else 120
        batch_size = args.batch_size if args.batch_size is not None else 64
        x_path = Path("data/X_tier2.npy")
        y_path = Path("data/y_tier2.npy")
        src_path = Path("data/sources.npy")
        cid_path = Path("data/clip_ids.npy")

        if not x_path.exists():
            print(f"\nERROR: Tier-2 kinematic features not found at {x_path}.")
            print("  Run first: python scripts/extract_features.py\n")
            return

    X = np.load(x_path)
    y = np.load(y_path)
    sources = np.load(src_path, allow_pickle=True) if src_path.exists() else np.array(["unknown"] * len(X))
    cids = np.load(cid_path) if cid_path.exists() else np.arange(len(X))

    # Filter dataset if specific subset requested
    if args.dataset != "all":
        mask = np.array([args.dataset.upper() in str(s).upper() for s in sources])
        X = X[mask]
        y = y[mask]
        sources = sources[mask]
        cids = cids[mask]
        print(f"  Filtered to dataset '{args.dataset}': {len(X)} samples remaining.")

    print(f"  Loaded Arrays      : X={X.shape}, y={y.shape}")
    print(f"  Class Distribution : Fight/Violence={int(y.sum())} ({y.sum()/len(y)*100:.1f}%), "
          f"NonFight/Normal={int((y==0).sum())} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Dataset Sources    : {', '.join(sorted(set(sources)))}")

    # 2. Stratified Split (80 / 10 / 10)
    X_train, y_train, X_val, y_val, X_test, y_test, src_test, cid_test = stratified_split(
        X, y, sources, clip_ids=cids, train_ratio=0.80, val_ratio=0.10, seed=42
    )
    print(f"\n  Stratified Split:")
    print(f"    Train : {len(X_train):>5} samples (Fight={int(y_train.sum())}, NonFight={int((y_train==0).sum())})")
    print(f"    Val   : {len(X_val):>5} samples (Fight={int(y_val.sum())}, NonFight={int((y_val==0).sum())})")
    print(f"    Test  : {len(X_test):>5} samples (Fight={int(y_test.sum())}, NonFight={int((y_test==0).sum())})")

    # 3. Train Selected Model
    if args.model == "vision_bilstm":
        ckpt_path = config.VISION_MODEL_PATH
        model = train_vision_classifier(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_path=ckpt_path,
            alpha=args.focal_alpha,
            target_recall=args.target_recall
        )
        # Get calibrated threshold
        thresh = getattr(model, 'calibrated_threshold', 0.50)
        evaluate_and_log(model, X_test, y_test, src_test, cid_test,
                         calibrated_thresh=thresh, model_name=f"vision_bilstm_{args.dataset}")

    elif args.model == "pose_gru":
        # Feature scaling for kinematic features
        scaler = FeatureScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
        scaler.save(SCALER_PATH)

        model = train_tier2(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
        evaluate_and_log(model, X_test_norm, y_test, src_test, cid_test,
                         calibrated_thresh=0.50, model_name=f"pose_gru_{args.dataset}")


if __name__ == "__main__":
    main()
    sys.exit(0)
