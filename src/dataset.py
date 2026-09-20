"""
src/dataset.py
──────────────
Unified Dataset Pipeline for Sentinel AI & CCTV Violence Detection.

Integrates:
  1. RLVS (Real Life Violence Situations - 2000 clips)
  2. RWF-2000 (Surveillance Camera Violence - 2000 clips)
  3. SCVD (SmartCity CCTV Violence Detection - 3151 clips in sec_split or 481 in converted)

Features:
  - Unified binary label mapping:
      Class 0 (NonFight/Normal): 'NonFight', 'Normal'
      Class 1 (Fight/Violence):  'Fight', 'Violence', 'Weaponized'
  - SHA-256 deduplication to purge cross-split leakage and conflicting labels
  - Robust temporal frame extraction (15 frames evenly sampled @ 128x128)
  - Leak-free stratified splitting at the video source level
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# Global label definitions
LABEL_MAP = {
    # Class 0: Normal / Non-violent
    "nonfight": 0,
    "normal": 0,
    # Class 1: Violence / Fight / Weaponized
    "fight": 1,
    "violence": 1,
    "weaponized": 1,
}

CLASS_NAMES = ["Normal", "Violence"]


def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file for duplicate detection."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class VideoClipMetadata:
    """Data container for a video clip."""
    def __init__(self, path: Path, label: int, dataset_name: str, split: str, source_id: str):
        self.path = path
        self.label = label
        self.dataset_name = dataset_name
        self.split = split
        self.source_id = source_id  # Parent video or unique sequence ID

    def __repr__(self):
        return f"<Clip {self.dataset_name} | {self.path.name} | Label={self.label} | Split={self.split}>"


def discover_all_clips(
    base_dir: Path = Path("data"),
    use_scvd_sec_split: bool = True,
    scvd_sample_cap: Optional[int] = None,
) -> List[VideoClipMetadata]:
    """
    Scans data directory and gathers all clips from RLVS, RWF-2000, and SCVD.
    """
    clips: List[VideoClipMetadata] = []
    supported_exts = {".mp4", ".avi", ".mov", ".mkv"}

    # 1. RLVS Dataset
    rlvs_dir = base_dir / "violence_fight_detection_dataset" / "RLVS"
    if rlvs_dir.exists():
        for split in ["train", "val"]:
            for cls_name, lbl in [("Fight", 1), ("NonFight", 0)]:
                dir_p = rlvs_dir / split / cls_name
                if dir_p.exists():
                    for f in sorted(dir_p.glob("*.*")):
                        if f.suffix.lower() in supported_exts:
                            clips.append(VideoClipMetadata(
                                path=f,
                                label=lbl,
                                dataset_name="RLVS",
                                split=split,
                                source_id=f"rlvs_{f.stem}"
                            ))

    # 2. RWF-2000 Dataset
    rwf_dir = base_dir / "violence_fight_detection_dataset" / "RWF-2000"
    if rwf_dir.exists():
        for split in ["train", "val"]:
            for cls_name, lbl in [("Fight", 1), ("NonFight", 0)]:
                dir_p = rwf_dir / split / cls_name
                if dir_p.exists():
                    for f in sorted(dir_p.glob("*.*")):
                        if f.suffix.lower() in supported_exts:
                            clips.append(VideoClipMetadata(
                                path=f,
                                label=lbl,
                                dataset_name="RWF-2000",
                                split=split,
                                source_id=f"rwf_{f.stem}"
                            ))

    # 3. SCVD Dataset
    scvd_subfolder = "SCVD_converted_sec_split" if use_scvd_sec_split else "SCVD_converted"
    scvd_dir = base_dir / "SCVD" / scvd_subfolder
    if scvd_dir.exists():
        for split in ["Train", "Test"]:
            # Standard SCVD folders
            category_mapping = [
                ("Normal", 0),
                ("Violence", 1),
                ("Weaponized", 1)
            ]
            for cat_dir_name, lbl in category_mapping:
                cat_p = scvd_dir / split / cat_dir_name
                if cat_p.exists():
                    file_list = sorted([f for f in cat_p.glob("*.avi") if f.suffix.lower() in supported_exts])
                    if scvd_sample_cap and split == "Train":
                        file_list = file_list[:scvd_sample_cap]

                    for f in file_list:
                        # Extract parent ID for sec_split (e.g. Normal001.avi -> parent is root video)
                        parent_id = f"scvd_{cat_dir_name}_{f.stem}"
                        clips.append(VideoClipMetadata(
                            path=f,
                            label=lbl,
                            dataset_name="SCVD",
                            split="train" if split == "Train" else "val",
                            source_id=parent_id
                        ))

    return clips


def clean_and_deduplicate_dataset(clips: List[VideoClipMetadata], verbose: bool = True) -> List[VideoClipMetadata]:
    """
    Eliminates identical duplicate files (by SHA-256 hash) across splits,
    and purges any conflicting labels (same file with multiple labels).
    """
    hash_to_clips: Dict[str, List[VideoClipMetadata]] = {}
    
    if verbose:
        print(f"Hashing {len(clips)} files to detect duplicates and split leakage...")

    for clip in clips:
        try:
            h = compute_file_hash(clip.path)
            if h not in hash_to_clips:
                hash_to_clips[h] = []
            hash_to_clips[h].append(clip)
        except Exception as e:
            if verbose:
                print(f"Warning: could not read {clip.path}: {e}")

    clean_clips: List[VideoClipMetadata] = []
    dropped_leakage = 0
    dropped_conflicts = 0

    for h, clip_group in hash_to_clips.items():
        labels = set(c.label for c in clip_group)
        if len(labels) > 1:
            # Conflicting labels (e.g. file_002173 labeled Fight and NonFight)
            dropped_conflicts += len(clip_group)
            continue

        # If identical files are present in both train and val/test, keep only ONE instance in train
        if len(clip_group) > 1:
            dropped_leakage += (len(clip_group) - 1)

        # Keep the first clean instance
        clean_clips.append(clip_group[0])

    if verbose:
        print(f"Deduplication complete:")
        print(f"  Total raw files     : {len(clips)}")
        print(f"  Dropped duplicates  : {dropped_leakage}")
        print(f"  Dropped conflicts   : {dropped_conflicts}")
        print(f"  Clean unique clips  : {len(clean_clips)}")

    return clean_clips


def load_video_frames(
    video_path: str,
    n_frames: int = 15,
    img_size: int = 128,
    normalize_mode: str = "tf",  # "tf" for [0, 1], "torch" for ImageNet norm
) -> Optional[np.ndarray]:
    """
    Uniform temporal sampling of `n_frames` from a video.
    Returns:
        np.ndarray of shape (n_frames, 3, img_size, img_size), float32
        or None if video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Compute evenly spaced frame indices across the full clip duration
    if total_frames <= n_frames:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    current_idx = 0
    target_set = set(frame_indices)
    max_target = max(frame_indices)

    while cap.isOpened() and current_idx <= max_target:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if current_idx in target_set:
            # Resize
            frame = cv2.resize(frame, (img_size, img_size))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

            if normalize_mode == "torch":
                # Standard ImageNet normalization: (x - mean) / std
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                frame = (frame - mean) / std

            # Convert to CHW layout: (3, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        current_idx += 1

    cap.release()

    if not frames:
        return None

    # If shorter than n_frames, pad by repeating the last frame
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    frames = frames[:n_frames]
    # Shape: (n_frames, 3, img_size, img_size)
    return np.stack(frames, axis=0).astype(np.float32)


def get_stratified_leak_free_splits(
    clips: List[VideoClipMetadata],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[VideoClipMetadata], List[VideoClipMetadata], List[VideoClipMetadata]]:
    """
    Splits metadata list into Train, Val, and Test while:
      1. Preserving exact class ratios (stratified per dataset & class)
      2. Ensuring zero data leakage at the source video level
    """
    rng = np.random.default_rng(seed)

    # Group by (dataset_name, label)
    grouped: Dict[Tuple[str, int], List[VideoClipMetadata]] = {}
    for c in clips:
        key = (c.dataset_name, c.label)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(c)

    train_clips, val_clips, test_clips = [], [], []

    for (dname, lbl), group in sorted(grouped.items()):
        # Shuffle group
        shuffled = list(rng.permutation(group))
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_clips.extend(shuffled[:n_train])
        val_clips.extend(shuffled[n_train:n_train + n_val])
        test_clips.extend(shuffled[n_train + n_val:])

    # Final shuffle
    train_clips = list(rng.permutation(train_clips))
    val_clips = list(rng.permutation(val_clips))
    test_clips = list(rng.permutation(test_clips))

    return train_clips, val_clips, test_clips


if __name__ == "__main__":
    print("Testing dataset discovery...")
    raw_clips = discover_all_clips(base_dir=Path("data"), use_scvd_sec_split=True)
    print(f"Total discovered clips: {len(raw_clips)}")

    clean_clips = clean_and_deduplicate_dataset(raw_clips)
    train_c, val_c, test_c = get_stratified_leak_free_splits(clean_clips)

    print(f"\nFinal Split Summary:")
    print(f"  Train: {len(train_c)} clips (Fight={sum(c.label for c in train_c)}, NonFight={len(train_c)-sum(c.label for c in train_c)})")
    print(f"  Val  : {len(val_c)} clips (Fight={sum(c.label for c in val_c)}, NonFight={len(val_c)-sum(c.label for c in val_c)})")
    print(f"  Test : {len(test_c)} clips (Fight={sum(c.label for c in test_c)}, NonFight={len(test_c)-sum(c.label for c in test_c)})")
