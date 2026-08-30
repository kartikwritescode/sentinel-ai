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
    Extracts a contiguous sequence of 30 frames of 24-D features from a clip.
    Ensures natural kinematic velocities without large frame skipping jumps.
    """
    engineer.reset()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return None

    # Pick contiguous start frame index (center window for fight/action clips)
    if total_frames >= FRAMES_PER_CLIP:
        start_frame = (total_frames - FRAMES_PER_CLIP) // 2
    else:
        start_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    feature_sequence = []
    while cap.isOpened() and len(feature_sequence) < FRAMES_PER_CLIP:
        success, frame = cap.read()
        if not success or frame is None:
            break

        # Detect people + pose on GPU in single pass
        persons = detector.detect_and_track(frame)
        feat = engineer.update(frame, persons)
        feature_sequence.append(feat)

    cap.release()

    if not feature_sequence:
        return None

    # If clip is slightly shorter than FRAMES_PER_CLIP, pad with last valid feature vector
    while len(feature_sequence) < FRAMES_PER_CLIP:
        feature_sequence.append(feature_sequence[-1].copy())

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

    chk_x = OUTPUT_DIR / "X_tier2_partial.npy"
    chk_y = OUTPUT_DIR / "y_tier2_partial.npy"
    chk_s = OUTPUT_DIR / "sources_partial.npy"

    X_list, y_list, src_list = [], [], []
    skipped = 0
    total   = len(clips)

    if chk_x.exists() and chk_y.exists() and chk_s.exists():
        try:
            X_list = list(np.load(chk_x))
            y_list = list(np.load(chk_y))
            src_list = list(np.load(chk_s, allow_pickle=True))
            print(f"Resuming from checkpoint with {len(X_list)} clips already extracted.")
        except Exception:
            pass

    start_idx = len(X_list)

    for i in range(start_idx, total):
        video_path, label, dataset_name = clips[i]
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"[{i+1}/{total}] ({(i+1)/total*100:.1f}%) Processing: {dataset_name} - {os.path.basename(video_path)}")

        seq = process_clip(video_path, detector, engineer)
        if seq is None:
            skipped += 1
            continue

        X_list.append(seq)
        y_list.append(label)
        src_list.append(dataset_name)

        # Checkpoint every 200 clips
        if len(X_list) % 200 == 0:
            np.save(chk_x, np.array(X_list, dtype=np.float32))
            np.save(chk_y, np.array(y_list, dtype=np.float32))
            np.save(chk_s, np.array(src_list, dtype=object))

    if not X_list:
        print("ERROR: All clips were skipped. Check dataset paths.")
        return

    X       = np.array(X_list,  dtype=np.float32)
    y       = np.array(y_list,  dtype=np.float32)
    sources = np.array(src_list, dtype=object)

    np.save(OUTPUT_DIR / "X_tier2.npy", X)
    np.save(OUTPUT_DIR / "y_tier2.npy", y)
    np.save(OUTPUT_DIR / "sources.npy", sources)

    # Clean up partial files
    for p in [chk_x, chk_y, chk_s]:
        if p.exists():
            try:
                os.remove(p)
            except Exception:
                pass

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
