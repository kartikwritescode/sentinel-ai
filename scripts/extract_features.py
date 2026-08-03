"""
Step 1 training.

Walks through ALL video clips from both datasets:
  - RLVS  (1000 Fight + 1000 NonFight .mp4 files)
  - RWF-2000  (1000 Fight + 1000 NonFight .avi files)

Runs the full detection → pose → feature-engineering pipeline on each
clip, then saves the resulting feature sequences + labels as .npy files.

Run from the project root:
    .\\venv\\Scripts\\python scripts/extract_features.py

Outputs (saved to data/):
    data/X_tier2.npy   shape: (num_clips, FEATURE_WINDOW_FRAMES, TIER2_INPUT_SIZE)
    data/y_tier2.npy   shape: (num_clips,)   0 = NonFight, 1 = Fight
    data/sources.npy   shape: (num_clips,)   dataset name per clip (for eval)
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import cv2
from pathlib import Path

import config
from src.detection import PersonDetector
from src.pose      import PoseEstimator
from src.features  import FeatureEngineer

# Dataset sources 
# Each entry is a dict with:
#   root   : Path to the dataset root folder
#   name   : Short label saved in sources.npy (useful for cross-dataset eval in Phase 5)
#   splits : sub-folders to include (add 'val' here because we re-split in train.py anyway)
#
# Both datasets share the same internal structure:
#   <root>/<split>/Fight/    ← .mp4 (RLVS) or .avi (RWF-2000)
#   <root>/<split>/NonFight/

BASE = Path("data/violence_fight_detection_dataset")

DATA_SOURCES = [
    {
        "root"  : BASE / "RLVS",
        "name"  : "RLVS",
        "splits": ["train", "val"],
    },
    {
        "root"  : BASE / "RWF-2000",
        "name"  : "RWF-2000",
        "splits": ["train", "val"],
    },
]

OUTPUT_DIR   = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}   # covers both datasets

# How many frames we sample from each clip —> must match FEATURE_WINDOW_FRAMES
FRAMES_PER_CLIP = config.FEATURE_WINDOW_FRAMES   # 30 by default


def gather_clips():
    """
    Walk every source in DATA_SOURCES and collect (path, label, dataset_name) tuples.

    Returns:
        clips: list of (video_path: str, label: int, dataset_name: str)
               label 1 = Fight, 0 = NonFight
    """
    clips = []

    for source in DATA_SOURCES:
        root   = source["root"]
        name   = source["name"]
        splits = source["splits"]

        if not root.exists():
            print(f"  [SKIP] {name} root not found: {root.resolve()}")
            continue

        source_fight    = 0
        source_nonfight = 0

        for split in splits:
            fight_dir    = root / split / "Fight"
            nonfight_dir = root / split / "NonFight"

            for p in fight_dir.glob("**/*"):
                if p.suffix.lower() in SUPPORTED_EXTS:
                    clips.append((str(p), 1, name))
                    source_fight += 1

            for p in nonfight_dir.glob("**/*"):
                if p.suffix.lower() in SUPPORTED_EXTS:
                    clips.append((str(p), 0, name))
                    source_nonfight += 1

        print(f"  {name:<12}  Fight: {source_fight:>5}   NonFight: {source_nonfight:>5}")

    total_fight    = sum(l for _, l, _ in clips)
    total_nonfight = len(clips) - total_fight
    print(f"  {'TOTAL':<12}  Fight: {total_fight:>5}   NonFight: {total_nonfight:>5}")
    print()
    return clips


def process_clip(video_path, detector, engineer):
    """
    Run the full pipeline on one clip (1000% GPU accelerated).
    """
    engineer.reset()   # clear history between clips — very important

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return None

    # Calculate target frame indices to sample
    step = max(total_frames // FRAMES_PER_CLIP, 1)
    target_indices = set(i * step for i in range(FRAMES_PER_CLIP))

    feature_sequence = []
    current_frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if current_frame_idx in target_indices:
            # 1. Detect + track + pose estimate in 1 single pass on CUDA GPU
            persons = detector.detect_and_track(frame)

            # 2. Compute feature vector for this frame
            feat = engineer.update(frame, persons)
            feature_sequence.append(feat)

            if len(feature_sequence) == FRAMES_PER_CLIP:
                break

        current_frame_idx += 1

    cap.release()

    if len(feature_sequence) < FRAMES_PER_CLIP:
        return None   # clip too short

    return np.stack(feature_sequence[:FRAMES_PER_CLIP], axis=0)


def main():
    print("Initialising GPU models...")
    detector  = PersonDetector()
    engineer  = FeatureEngineer()
    print("GPU Models ready.\n")

    print("Scanning datasets:")
    clips = gather_clips()   # list of (path, label, dataset_name)
    if not clips:
        print("ERROR: No video clips found. Check DATA_SOURCES paths above.")
        return

    X_list, y_list, src_list = [], [], []
    skipped = 0
    total   = len(clips)

    for i, (video_path, label, dataset_name) in enumerate(clips):
        # \r overwrites the same line each iteration — shows rolling progress
        print(f"[{i+1}/{total}]  {dataset_name:<12}  {os.path.basename(video_path):<40}", end="\r")

        seq = process_clip(video_path, detector, engineer)
        if seq is None:
            skipped += 1
            continue

        X_list.append(seq)
        y_list.append(label)
        src_list.append(dataset_name)

    print()  # newline after \r progress

    if not X_list:
        print("ERROR: All clips were skipped. Check dataset paths.")
        return

    X       = np.array(X_list,  dtype=np.float32)   # (N, frames, features)
    y       = np.array(y_list,  dtype=np.float32)   # (N,)
    sources = np.array(src_list, dtype=object)       # (N,)  string array

    np.save(OUTPUT_DIR / "X_tier2.npy", X)
    np.save(OUTPUT_DIR / "y_tier2.npy", y)
    np.save(OUTPUT_DIR / "sources.npy", sources)

    # Final summary 
    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"{'='*55}")
    print(f"  Total processed : {len(X_list):>5}  |  Skipped: {skipped}")
    print()

    # Per-dataset breakdown
    unique_sources = sorted(set(src_list))
    for src in unique_sources:
        mask         = sources == src
        n_fight      = int((y[mask] == 1).sum())
        n_nonfight   = int((y[mask] == 0).sum())
        print(f"  {src:<14}  Fight: {n_fight:>5}   NonFight: {n_nonfight:>5}")

    print()
    print(f"  X shape  : {X.shape}   →  data/X_tier2.npy")
    print(f"  y shape  : {y.shape}   →  data/y_tier2.npy")
    print(f"  sources  : {sources.shape}   →  data/sources.npy")
    print(f"  Feature size per frame: {X.shape[2]}")

    if X.shape[2] != config.TIER2_INPUT_SIZE:
        print(f"\n  !! MISMATCH: X has {X.shape[2]} features, config.TIER2_INPUT_SIZE={config.TIER2_INPUT_SIZE}")
        print(f"  Fix: set TIER2_INPUT_SIZE = {X.shape[2]} in config.py, then run train.py.")
    else:
        print(f"\n  Feature size matches config ({config.TIER2_INPUT_SIZE}). Run: python scripts/train.py")


if __name__ == "__main__":
    main()
