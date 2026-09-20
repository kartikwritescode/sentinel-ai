"""
test.py
───────
Sentinel AI Diagnostic & End-to-End Self-Test Script.

Verifies:
  1. CUDA GPU and PyTorch environment
  2. Dataset directory structures (SCVD, RLVS, RWF-2000)
  3. Feature caches (data/X_vision.npy, data/X_tier2.npy)
  4. Model checkpoints (models/vision_bilstm.pt, models/tier2_gru.pt)
  5. Real-time UnifiedInferencer pipeline with synthetic video frames

Usage:
  python test.py
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

import config
from src.classifier import UnifiedInferencer, VisionBiLSTMClassifier, SuspiciousActivityClassifier


def main():
    print("\n" + "=" * 65)
    print("  SENTINEL AI -- SYSTEM DIAGNOSTIC & PIPELINE TEST")
    print("=" * 65)

    # 1. Environment & CUDA
    print("\n[1/5] Environment & Hardware Check:")
    print(f"  Python Version   : {sys.version.split()[0]}")
    print(f"  PyTorch Version  : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available   : {cuda_avail}")
    if cuda_avail:
        print(f"  GPU Device       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total       : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Dataset Directories
    print("\n[2/5] Dataset Verification:")
    datasets = {
        "SCVD Split": Path("data/SCVD/SCVD_converted_sec_split"),
        "SCVD Converted": Path("data/SCVD/SCVD_converted"),
        "RLVS": Path("data/violence_fight_detection_dataset/RLVS"),
        "RWF-2000": Path("data/violence_fight_detection_dataset/RWF-2000"),
    }
    for name, p in datasets.items():
        exists = p.exists()
        count = len(list(p.rglob("*.*"))) if exists else 0
        status = f"OK ({count} files)" if exists and count > 0 else "MISSING / EMPTY"
        print(f"  {name:<16}: {status} -> {p}")

    # 3. Feature Arrays
    print("\n[3/5] Feature Cache Verification:")
    features = {
        "Vision Arrays": [
            Path("data/X_vision.npy"),
            Path("data/y_vision.npy"),
            Path("data/sources_vision.npy")
        ],
        "Vision Partial": [
            Path("data/X_vision_partial.npy")
        ],
        "Kinematic Arrays": [
            Path("data/X_tier2.npy"),
            Path("data/y_tier2.npy")
        ]
    }
    for group, paths in features.items():
        print(f"  {group}:")
        for p in paths:
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"    - {p.name:<24}: FOUND ({size_mb:.2f} MB)")
            else:
                print(f"    - {p.name:<24}: NOT FOUND (will be generated)")

    # 4. Model Checkpoints
    print("\n[4/5] Model Checkpoint Verification:")
    models = {
        "Vision BiLSTM": Path(config.VISION_MODEL_PATH),
        "Kinematic Pose GRU": Path(config.TIER2_MODEL_PATH),
        "Legacy MoBiLSTM": Path("models/MoBiLSTM_model.h5")
    }
    for name, p in models.items():
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {name:<20}: FOUND ({size_mb:.2f} MB) -> {p}")
        else:
            print(f"  {name:<20}: NOT YET TRAINED -> {p}")

    # 5. Live Inferencer Pipeline Simulation
    print("\n[5/5] UnifiedInferencer Smoke Test:")
    try:
        inferencer = UnifiedInferencer()
        print(f"  Model Type Loaded: {inferencer.model_type}")

        # Simulate pushing 20 synthetic CCTV video frames (1280x720 RGB)
        print("  Pushing 20 test frames (1280x720x3) through inference pipeline...")
        t0 = time.time()
        last_prob = 0.0
        for i in range(20):
            dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            prob = inferencer.push_frame(dummy_frame)
            if prob is not None:
                last_prob = prob
        elapsed = time.time() - t0
        fps = 20.0 / elapsed if elapsed > 0 else 0
        print(f"  [PASSED] Ingested 20 frames in {elapsed*1000:.1f}ms ({fps:.1f} FPS)")
        print(f"  Inferencer Output: Latest Violence Probability = {last_prob:.4f}")
    except Exception as e:
        print(f"  [WARNING] Inferencer test encountered: {e}")

    print("\n" + "=" * 65)
    print("  DIAGNOSTIC TEST COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
