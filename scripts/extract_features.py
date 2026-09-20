"""
Step 1 training — Multi-Crop Feature Extraction.

Walks through ALL video clips from both datasets:
  - RLVS  (1000 Fight + 1000 NonFight .mp4 files)
  - RWF-2000  (1000 Fight + 1000 NonFight .avi files)

Runs the full detection → pose → feature-engineering pipeline on each
clip, extracting MULTIPLE temporal crops per clip to triple the dataset.

Run from the project root:
    .\\venv\\Scripts\\python scripts/extract_features.py

Outputs (saved to data/):
    data/X_tier2.npy    shape: (num_samples, FEATURE_WINDOW_FRAMES, TIER2_INPUT_SIZE)
    data/y_tier2.npy    shape: (num_samples,)   0 = NonFight, 1 = Fight
    data/sources.npy    shape: (num_samples,)   dataset name per sample
    data/clip_ids.npy   shape: (num_samples,)   clip index for clip-aware splitting
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

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

FRAMES_PER_CLIP = config.FEATURE_WINDOW_FRAMES   # 30 by default

# Multi-crop: extract up to 3 temporal windows per clip
MAX_CROPS_PER_CLIP = 3


def gather_clips():
    """
    Walk every source in DATA_SOURCES and collect (path, label, dataset_name) tuples.
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


def _get_crop_start_positions(total_frames, window_size, max_crops):
    """
    Compute start frame positions for multi-crop extraction.

    For clips long enough, extracts 3 crops at 25%, 50%, 75% positions.
    For shorter clips, extracts fewer crops.
    All positions are guaranteed non-negative and valid.
    """
    if total_frames < window_size:
        return [0]  # single crop, will be padded

    available = total_frames - window_size

    if available == 0:
        return [0]

    if max_crops >= 3 and available >= 2:
        # 3 crops at 25%, 50%, 75%
        positions = [
            available // 4,           # ~25%
            available // 2,           # ~50% (center)
            (3 * available) // 4,     # ~75%
        ]
        # Deduplicate (for short clips where positions overlap)
        positions = sorted(set(positions))
        return positions
    elif max_crops >= 2 and available >= 1:
        return [0, available]
    else:
        return [available // 2]  # center


def process_clip_multicrop(video_path, detector, engineer):
    """
    Extracts MULTIPLE 30-frame feature sequences from a single clip.
    Each crop starts at a different temporal position.
    The FeatureEngineer is reset between crops to ensure independent velocity computation.

    Returns:
        list of np.ndarray, each shape (FRAMES_PER_CLIP, feat_dim), or empty list
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return []

    start_positions = _get_crop_start_positions(total_frames, FRAMES_PER_CLIP, MAX_CROPS_PER_CLIP)

    crops = []
    for start_frame in start_positions:
        engineer.reset()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        feature_sequence = []
        while cap.isOpened() and len(feature_sequence) < FRAMES_PER_CLIP:
            success, frame = cap.read()
            if not success or frame is None:
                break

            persons = detector.detect_and_track(frame)
            feat = engineer.update(frame, persons)
            feature_sequence.append(feat)

        if not feature_sequence:
            continue

        # Pad short sequences by repeating last valid feature vector
        while len(feature_sequence) < FRAMES_PER_CLIP:
            feature_sequence.append(feature_sequence[-1].copy())

        crops.append(np.stack(feature_sequence[:FRAMES_PER_CLIP], axis=0))

    cap.release()
    return crops


def main():
    print("Initialising GPU models...")
    detector  = PersonDetector()
    engineer  = FeatureEngineer()
    print("GPU Models ready.\n")

    print("Scanning datasets:")
    clips = gather_clips()
    if not clips:
        print("ERROR: No video clips found. Check DATA_SOURCES paths above.")
        return

    chk_x = OUTPUT_DIR / "X_tier2_partial.npy"
    chk_y = OUTPUT_DIR / "y_tier2_partial.npy"
    chk_s = OUTPUT_DIR / "sources_partial.npy"
    chk_c = OUTPUT_DIR / "clip_ids_partial.npy"

    X_list, y_list, src_list, clip_id_list = [], [], [], []
    skipped = 0
    total   = len(clips)

    # Try to resume from checkpoint
    start_clip_idx = 0
    if chk_x.exists() and chk_y.exists() and chk_s.exists() and chk_c.exists():
        try:
            X_list = list(np.load(chk_x))
            y_list = list(np.load(chk_y))
            src_list = list(np.load(chk_s, allow_pickle=True))
            clip_id_list = list(np.load(chk_c))
            # Determine which clip to resume from
            if clip_id_list:
                start_clip_idx = int(max(clip_id_list)) + 1
            print(f"Resuming from checkpoint: {len(X_list)} samples from {start_clip_idx} clips.")
        except Exception:
            X_list, y_list, src_list, clip_id_list = [], [], [], []
            start_clip_idx = 0

    for i in range(start_clip_idx, total):
        video_path, label, dataset_name = clips[i]
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"[{i+1}/{total}] ({(i+1)/total*100:.1f}%) Processing: "
                  f"{dataset_name} - {os.path.basename(video_path)} | Samples: {len(X_list)}")

        crop_seqs = process_clip_multicrop(video_path, detector, engineer)
        if not crop_seqs:
            skipped += 1
            continue

        for seq in crop_seqs:
            X_list.append(seq)
            y_list.append(label)
            src_list.append(dataset_name)
            clip_id_list.append(i)

        # Checkpoint every 200 clips
        if (i + 1) % 200 == 0:
            np.save(chk_x, np.array(X_list, dtype=np.float32))
            np.save(chk_y, np.array(y_list, dtype=np.float32))
            np.save(chk_s, np.array(src_list, dtype=object))
            np.save(chk_c, np.array(clip_id_list, dtype=np.int32))

    if not X_list:
        print("ERROR: All clips were skipped. Check dataset paths.")
        return

    X        = np.array(X_list,      dtype=np.float32)
    y        = np.array(y_list,      dtype=np.float32)
    sources  = np.array(src_list,    dtype=object)
    clip_ids = np.array(clip_id_list, dtype=np.int32)

    np.save(OUTPUT_DIR / "X_tier2.npy", X)
    np.save(OUTPUT_DIR / "y_tier2.npy", y)
    np.save(OUTPUT_DIR / "sources.npy", sources)
    np.save(OUTPUT_DIR / "clip_ids.npy", clip_ids)

    # Clean up partial files
    for p in [chk_x, chk_y, chk_s, chk_c]:
        if p.exists():
            try:
                os.remove(p)
            except Exception:
                pass

    # Final summary
    n_clips = len(set(clip_id_list))
    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"{'='*55}")
    print(f"  Clips processed  : {n_clips:>5}  |  Skipped: {skipped}")
    print(f"  Total samples    : {len(X_list):>5}  ({len(X_list)/max(n_clips,1):.1f} crops/clip avg)")
    print()

    # Per-dataset breakdown
    unique_sources = sorted(set(src_list))
    for src in unique_sources:
        mask         = sources == src
        n_fight      = int((y[mask] == 1).sum())
        n_nonfight   = int((y[mask] == 0).sum())
        n_clips_src  = len(set(clip_ids[mask]))
        print(f"  {src:<14}  Fight: {n_fight:>5}   NonFight: {n_nonfight:>5}  ({n_clips_src} clips)")

    print()
    print(f"  X shape   : {X.shape}   →  data/X_tier2.npy")
    print(f"  y shape   : {y.shape}   →  data/y_tier2.npy")
    print(f"  sources   : {sources.shape}   →  data/sources.npy")
    print(f"  clip_ids  : {clip_ids.shape}   →  data/clip_ids.npy")
    print(f"  Feature size per frame: {X.shape[2]}")

    if X.shape[2] != config.TIER2_INPUT_SIZE:
        print(f"\n  !! MISMATCH: X has {X.shape[2]} features, config.TIER2_INPUT_SIZE={config.TIER2_INPUT_SIZE}")
        print(f"  Fix: set TIER2_INPUT_SIZE = {X.shape[2]} in config.py, then run train.py.")
    else:
        print(f"\n  Feature size matches config ({config.TIER2_INPUT_SIZE}). Run: python scripts/train.py")


if __name__ == "__main__":
    main()
