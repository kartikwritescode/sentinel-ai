"""
scripts/evaluate.py
───────────────────
Comprehensive Evaluation for Conv-BiGRU Temporal Attention Classifier.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time
import json
import torch
import numpy as np
from pathlib import Path

import config
from src.classifier import SuspiciousActivityGRU, FeatureScaler, SCALER_PATH
from scripts.train import stratified_split


def evaluate_model():
    x_path   = Path("data/X_tier2.npy")
    y_path   = Path("data/y_tier2.npy")
    src_path = Path("data/sources.npy")
    model_path = Path(config.TIER2_MODEL_PATH)

    if not x_path.exists() or not y_path.exists():
        print("ERROR: Feature data not found. Run scripts/extract_features.py first.")
        return

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path}. Run scripts/train.py first.")
        return

    X       = np.load(x_path)
    y       = np.load(y_path)
    sources = np.load(src_path, allow_pickle=True) if src_path.exists() else np.array(["unknown"] * len(X))

    # Perform stratified split (same split used in train.py)
    _, _, X_val, y_val, X_test, y_test, src_test = stratified_split(
        X, y, sources, train_ratio=0.80, val_ratio=0.10, seed=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load FeatureScaler
    scaler = FeatureScaler()
    if not scaler.load(SCALER_PATH):
        scaler.fit(X)

    X_val_norm  = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    # Load model
    model = SuspiciousActivityGRU().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Latency benchmark
    X_t = torch.FloatTensor(X_test_norm).to(device)
    with torch.no_grad():
        _ = model(X_t[:10])  # Warmup
        t0 = time.time()
        logits = model(X_t).squeeze(1)
        probs = logits.cpu().numpy()
        total_time = time.time() - t0

    latency_ms = (total_time / len(X_test)) * 1000.0
    throughput_fps = len(X_test) / total_time

    # Find optimal threshold on validation set
    with torch.no_grad():
        val_probs = model(torch.FloatTensor(X_val_norm).to(device)).squeeze(1).cpu().numpy()

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

    # Predictions on test set using optimal threshold
    preds = (probs >= best_thresh).astype(int)
    targets = y_test.astype(int)

    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    total = len(targets)
    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score    = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Per-dataset breakdown
    dataset_metrics = {}
    for src in sorted(set(src_test)):
        mask = (src_test == src)
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        sub_correct = np.sum(sub_preds == sub_targets)
        sub_total = len(sub_targets)
        sub_acc = sub_correct / sub_total * 100.0 if sub_total > 0 else 0.0
        dataset_metrics[str(src)] = {
            "total_samples": int(sub_total),
            "correct": int(sub_correct),
            "accuracy_pct": round(sub_acc, 2)
        }

    metrics_report = {
        "device": str(device),
        "test_samples": total,
        "calibrated_threshold": round(best_thresh, 2),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        },
        "metrics": {
            "accuracy_pct": round(accuracy * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "recall_sensitivity_pct": round(recall * 100.0, 2),
            "specificity_pct": round(specificity * 100.0, 2),
            "f1_score_pct": round(f1_score * 100.0, 2),
            "false_positive_rate_pct": round(fpr * 100.0, 2)
        },
        "latency": {
            "avg_inference_ms": round(latency_ms, 3),
            "throughput_fps": round(throughput_fps, 1)
        },
        "per_dataset_accuracy": dataset_metrics
    }

    report_path = Path("data/eval_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=4)

    print("\n=======================================================")
    print("      CONV-BIGRU ATTENTION MODEL EVALUATION REPORT     ")
    print("=======================================================")
    print(f" Calibrated Threshold: {best_thresh:.2f}")
    print(f" Test Set Size       : {total} clips")
    print(" -------------------------------------------------------")
    print(f" Accuracy            : {accuracy * 100.0:.2f}%")
    print(f" Precision           : {precision * 100.0:.2f}%")
    print(f" Recall (Sensitivity): {recall * 100.0:.2f}%")
    print(f" Specificity         : {specificity * 100.0:.2f}%")
    print(f" F1-Score            : {f1_score * 100.0:.2f}%")
    print(f" False Positive Rate : {fpr * 100.0:.2f}%")
    print(" -------------------------------------------------------")
    print(" CONFUSION MATRIX:")
    print(f"   TP (Fight detected as Fight)      : {tp:>4}")
    print(f"   FP (NonFight detected as Fight)   : {fp:>4}")
    print(f"   TN (NonFight detected as NonFight): {tn:>4}")
    print(f"   FN (Fight detected as NonFight)   : {fn:>4}")
    print(" -------------------------------------------------------")
    print(" PER-DATASET ACCURACY:")
    for src, d in dataset_metrics.items():
        print(f"   {src:<14} : {d['accuracy_pct']:.2f}% ({d['correct']}/{d['total_samples']})")
    print(" -------------------------------------------------------")
    print(f" Avg Latency per Window : {latency_ms:.3f} ms")
    print(f" Throughput             : {throughput_fps:.1f} FPS")
    print("=======================================================")
    print(f"Saved full JSON report to: {report_path.resolve()}\n")


if __name__ == "__main__":
    evaluate_model()

