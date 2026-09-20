"""
scripts/evaluate.py
───────────────────
Comprehensive Model Evaluation & Benchmarking for Sentinel AI.

Evaluates either:
  - Vision-BiLSTM Model (--model-type vision_bilstm)
  - Kinematic Pose Model (--model-type pose_gru)

Reports:
  - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, Macro F1, Weighted F1, False Positive Rate
  - Confusion Matrix (TP, FP, TN, FN)
  - Per-dataset performance breakdown (RLVS, RWF-2000, SCVD)
  - Hardware Latency (ms/window) & Throughput (FPS)
  - Saves full report to data/eval_report.json
"""

import sys
import os
import argparse
import time
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
    SCALER_PATH
)
from scripts.train import stratified_split


def evaluate(
    model_type: str = "vision_bilstm",
    model_path_str: str = None,
    dataset_filter: str = "all"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"\n[Evaluator] Device: {device} " +
          (f"({torch.cuda.get_device_name(0)}) | FP16 / TF32 Enabled" if device.type == 'cuda' else 'CPU'))

    # Determine paths
    if model_type == "vision_bilstm":
        x_path = Path("data/X_vision.npy")
        y_path = Path("data/y_vision.npy")
        src_path = Path("data/sources_vision.npy")
        cid_path = Path("data/clip_ids_vision.npy")
        default_model_path = config.VISION_MODEL_PATH
    else:
        x_path = Path("data/X_tier2.npy")
        y_path = Path("data/y_tier2.npy")
        src_path = Path("data/sources.npy")
        cid_path = Path("data/clip_ids.npy")
        default_model_path = config.TIER2_MODEL_PATH

    m_path = Path(model_path_str if model_path_str else default_model_path)
    if not m_path.exists():
        print(f"ERROR: Model checkpoint not found at: {m_path}")
        return

    if not x_path.exists():
        print(f"ERROR: Feature array not found at: {x_path}")
        return

    # Load data
    X = np.load(x_path)
    y = np.load(y_path)
    sources = np.load(src_path, allow_pickle=True) if src_path.exists() else np.array(["unknown"] * len(X))
    cids = np.load(cid_path) if cid_path.exists() else np.arange(len(X))

    if dataset_filter != "all":
        mask = np.array([dataset_filter.upper() in str(s).upper() for s in sources])
        X = X[mask]
        y = y[mask]
        sources = sources[mask]
        cids = cids[mask]
        print(f"[Evaluator] Filtered to dataset '{dataset_filter}': {len(X)} samples.")

    # Stratified test split (identical seed=42 to training)
    _, _, X_val, y_val, X_test, y_test, src_test, cid_test = stratified_split(
        X, y, sources, clip_ids=cids, train_ratio=0.80, val_ratio=0.10, seed=42
    )

    # Load Model
    calibrated_thresh = 0.50
    if model_type == "vision_bilstm":
        model = VisionBiLSTMClassifier().to(device)
        ckpt = torch.load(m_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            calibrated_thresh = float(ckpt.get('calibrated_threshold', 0.50))
        else:
            model.load_state_dict(ckpt)
    else:
        scaler = FeatureScaler()
        if os.path.exists(SCALER_PATH):
            scaler.load(SCALER_PATH)
        else:
            scaler.fit(X)
        X_test = scaler.transform(X_test)
        model = SuspiciousActivityClassifier().to(device)
        model.load_state_dict(torch.load(m_path, map_location=device, weights_only=False))

    model.eval()
    print(f"[Evaluator] Loaded checkpoint: {m_path.name} (Calibrated Threshold: {calibrated_thresh:.2f})")

    # Latency & Throughput Benchmark
    X_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        # Warmup
        _ = model(X_t[:min(10, len(X_t))])
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()
        probs = model(X_t).squeeze(1).cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_eval_time = time.time() - t0

    latency_ms = (total_eval_time / len(X_test)) * 1000.0
    throughput_fps = len(X_test) / total_eval_time if total_eval_time > 0 else 0.0

    # Predictions
    preds = (probs >= calibrated_thresh).astype(int)
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
    print(f"  MODEL EVALUATION REPORT: {model_type.upper()}")
    print(f"{'='*65}")
    print(f" Checkpoint          : {m_path.resolve()}")
    print(f" Calibrated Threshold: {calibrated_thresh:.2f}")
    print(f" Test Set Samples    : {total}")
    print(f" -------------------------------------------------------------")
    print(f" Accuracy            : {acc:.2f}%")
    print(f" Macro F1-Score      : {macro_f1:.2f}%")
    print(f" Weighted F1-Score   : {weighted_f1:.2f}%")
    print(f" Precision           : {prec:.2f}%")
    print(f" Recall (Sensitivity): {rec:.2f}%")
    print(f" Specificity         : {spec:.2f}%")
    print(f" False Positive Rate : {fpr:.2f}%")
    print(f" -------------------------------------------------------------")
    print(f" CONFUSION MATRIX:")
    print(f"   TP (Fight as Fight)       : {tp:>5}")
    print(f"   FP (NonFight as Fight)    : {fp:>5}")
    print(f"   TN (NonFight as NonFight) : {tn:>5}")
    print(f"   FN (Fight as NonFight)    : {fn:>5}")
    print(f" -------------------------------------------------------------")
    print(f" PER-DATASET PERFORMANCE:")
    dataset_metrics = {}
    for src in sorted(set(src_test)):
        mask = (src_test == src)
        sub_correct = np.sum(preds[mask] == targets[mask])
        sub_total = np.sum(mask)
        sub_acc = sub_correct / sub_total * 100.0 if sub_total > 0 else 0.0
        sub_f1 = f1_score(targets[mask], preds[mask], average="macro", zero_division=0) * 100.0
        print(f"   {str(src):<14} : {sub_acc:6.2f}% (Acc) | {sub_f1:6.2f}% (Macro F1) | {sub_correct}/{sub_total} clips")
        dataset_metrics[str(src)] = {
            "total": int(sub_total),
            "correct": int(sub_correct),
            "accuracy_pct": round(sub_acc, 2),
            "macro_f1_pct": round(sub_f1, 2)
        }
    print(f" -------------------------------------------------------------")
    print(f" Hardware Latency    : {latency_ms:.3f} ms per window")
    print(f" Throughput          : {throughput_fps:,.1f} FPS")
    print(f"{'='*65}")

    report = {
        "model_type": model_type,
        "checkpoint": str(m_path.resolve()),
        "device": str(device),
        "test_samples": total,
        "calibrated_threshold": round(calibrated_thresh, 2),
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
        "per_dataset": dataset_metrics,
        "latency": {
            "avg_latency_ms": round(latency_ms, 3),
            "throughput_fps": round(throughput_fps, 1)
        }
    }

    report_path = Path("data/eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved full JSON evaluation report to: {report_path.resolve()}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentinel AI Model Checkpoint")
    parser.add_argument("--model-type", type=str, default="vision_bilstm",
                        choices=["vision_bilstm", "pose_gru"],
                        help="Model architecture type: 'vision_bilstm' or 'pose_gru'")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "scvd", "rlvs", "rwf"],
                        help="Evaluate on subset or 'all'")
    args = parser.parse_args()

    evaluate(model_type=args.model_type, model_path_str=args.model_path, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
