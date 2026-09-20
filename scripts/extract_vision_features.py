"""
scripts/extract_vision_features.py
──────────────────────────────────
High-Speed GPU Vision Feature Extraction using ImageNet-Pretrained MobileNetV2.

Replicates and extends the feature extraction strategy from cctv-classification.ipynb:
  1. Reads video clips from Unified Dataset (RLVS, RWF-2000, SCVD).
  2. Samples 15 evenly-spaced frames per clip, resized to 128x128, normalized.
  3. Passes frames through PyTorch MobileNetV2 convolutional backbone.
  4. Applies Global Average Pooling to produce 1280-D spatial embeddings per frame.
  5. Checkpoints progress to disk.

Outputs:
  data/X_vision.npy           shape: (num_clips, 15, 1280)
  data/y_vision.npy           shape: (num_clips,)
  data/sources_vision.npy     shape: (num_clips,)
  data/clip_ids_vision.npy    shape: (num_clips,)
"""

import sys
import os
import argparse
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import concurrent.futures
import config
from src.dataset import (
    discover_all_clips,
    clean_and_deduplicate_dataset,
    load_video_frames,
    VideoClipMetadata
)


class VisionBackboneExtractor(nn.Module):
    """
    MobileNetV2 feature extractor matching the notebook's TimeDistributed(MobileNetV2)
    + GlobalAveragePooling2D architecture, accelerated with PyTorch AMP FP16.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(device)
        self.eval()

    @torch.no_grad()
    def extract_batch(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames_tensor: (batch_size, n_frames, 3, H, W)
        Returns:
            (batch_size, n_frames, 1280)
        """
        B, T, C, H, W = frames_tensor.shape
        flat_frames = frames_tensor.view(B * T, C, H, W)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == "cuda" else torch.float32):
            feat_maps = self.features(flat_frames)  # (B*T, 1280, H/32, W/32)
            pooled = self.pool(feat_maps).flatten(1)  # (B*T, 1280)
        return pooled.view(B, T, 1280).float()


def _decode_clip_task(task_args):
    clip, idx, n_frames, img_size = task_args
    frames = load_video_frames(
        clip.path,
        n_frames=n_frames,
        img_size=img_size,
        normalize_mode="tf"
    )
    if frames is None:
        return None
    return (frames, clip.label, clip.dataset_name, idx, str(clip.path))


def main():
    parser = argparse.ArgumentParser(description="Extract MobileNetV2 Vision Features (GPU-Accelerated)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "scvd", "rlvs", "rwf"],
                        help="Which dataset to extract: 'all', 'scvd', 'rlvs', 'rwf'")
    parser.add_argument("--batch-size", type=int, default=64, help="Clips per extraction batch (default: 64)")
    parser.add_argument("--img-size", type=int, default=128, help="Frame height/width (default: 128)")
    parser.add_argument("--frame-count", type=int, default=15, help="Frames per clip (default: 15)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4),
                        help="CPU worker threads for parallel video decoding (default: 12)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'='*65}")
    print(f"  VISION FEATURE EXTRACTION (MAX GPU/CPU PIPELINE)")
    print(f"{'='*65}")
    print(f"  Device           : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  CuDNN Benchmark  : Enabled")
    print(f"  FP16 AMP / TF32  : Enabled")
    print(f"  CPU Worker Pool  : {args.workers} threads")
    print(f"  GPU Batch Size   : {args.batch_size} clips ({args.batch_size * args.frame_count} frames/batch)")

    extractor = VisionBackboneExtractor(device)
    print(f"  MobileNetV2 ImageNet backbone loaded successfully.\n")

    # 1. Discover and clean clips
    print("[Vision Extractor] Scanning dataset directories...")
    all_clips = discover_all_clips(base_dir=Path("data"), use_scvd_sec_split=True)
    clean_clips = clean_and_deduplicate_dataset(all_clips)

    # Filter if single dataset requested
    if args.dataset != "all":
        target = args.dataset.upper()
        clean_clips = [c for c in clean_clips if target in c.dataset_name.upper()]
        print(f"[Vision Extractor] Filtered to '{args.dataset}': {len(clean_clips)} clips")

    output_dir = Path("data")
    prefix = f"vision_{args.dataset}" if args.dataset != "all" else "vision"

    chk_x = output_dir / f"X_{prefix}_partial.npy"
    chk_y = output_dir / f"y_{prefix}_partial.npy"
    chk_s = output_dir / f"sources_{prefix}_partial.npy"
    chk_c = output_dir / f"clip_ids_{prefix}_partial.npy"
    chk_p = output_dir / f"paths_{prefix}_partial.npy"

    X_list, y_list, src_list, cid_list, path_list = [], [], [], [], []
    start_idx = 0

    if chk_x.exists() and chk_y.exists() and chk_s.exists() and chk_c.exists():
        try:
            X_list = list(np.load(chk_x))
            y_list = list(np.load(chk_y))
            src_list = list(np.load(chk_s, allow_pickle=True))
            cid_list = list(np.load(chk_c))
            path_list = list(np.load(chk_p, allow_pickle=True))
            start_idx = len(X_list)
            print(f"[Vision Extractor] Resuming from checkpoint: {start_idx} clips already extracted.")
        except Exception as e:
            print(f"[Vision Extractor] Checkpoint load failed ({e}), restarting.")
            X_list, y_list, src_list, cid_list, path_list = [], [], [], [], []
            start_idx = 0

    total = len(clean_clips)
    t0 = time.time()

    print(f"[Vision Extractor] Starting parallel extraction from clip {start_idx}/{total}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for chunk_start in range(start_idx, total, args.batch_size):
            chunk_end = min(chunk_start + args.batch_size, total)
            chunk_tasks = [
                (clean_clips[k], k, args.frame_count, args.img_size)
                for k in range(chunk_start, chunk_end)
            ]

            # Parallel decode across CPU threads
            results = list(executor.map(_decode_clip_task, chunk_tasks))
            valid = [r for r in results if r is not None]

            if not valid:
                continue

            # Batch tensors into pinned memory and transfer to GPU
            batch_frames = np.array([r[0] for r in valid], dtype=np.float32)
            tensor_batch = torch.from_numpy(batch_frames)
            if device.type == "cuda":
                tensor_batch = tensor_batch.pin_memory().to(device, non_blocking=True)
            else:
                tensor_batch = tensor_batch.to(device)

            with torch.no_grad():
                feats = extractor.extract_batch(tensor_batch).cpu().numpy()

            for j, feat in enumerate(feats):
                _, lbl, s_name, c_id, v_path = valid[j]
                X_list.append(feat)
                y_list.append(lbl)
                src_list.append(s_name)
                cid_list.append(c_id)
                path_list.append(v_path)

            elapsed = time.time() - t0
            speed = len(X_list) / elapsed if elapsed > 0 else 0
            pct = (len(X_list) / total) * 100.0
            last_clip = clean_clips[chunk_end - 1]
            print(f"  [{len(X_list):>5}/{total}] ({pct:5.1f}%) "
                  f"Extracted: {last_clip.dataset_name:<8} - {last_clip.path.name:<25} | {speed:5.1f} clips/s",
                  flush=True)

            # Checkpoint every ~500 clips
            if len(X_list) % 512 < args.batch_size or chunk_end == total:
                np.save(chk_x, np.array(X_list, dtype=np.float32))
                np.save(chk_y, np.array(y_list, dtype=np.float32))
                np.save(chk_s, np.array(src_list, dtype=object))
                np.save(chk_c, np.array(cid_list, dtype=np.int32))
                np.save(chk_p, np.array(path_list, dtype=object))

    # Save final arrays
    final_x = output_dir / f"X_{prefix}.npy"
    final_y = output_dir / f"y_{prefix}.npy"
    final_s = output_dir / f"sources_{prefix}.npy"
    final_c = output_dir / f"clip_ids_{prefix}.npy"
    final_p = output_dir / f"paths_{prefix}.npy"

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    sources = np.array(src_list, dtype=object)
    cids = np.array(cid_list, dtype=np.int32)
    paths = np.array(path_list, dtype=object)

    np.save(final_x, X)
    np.save(final_y, y)
    np.save(final_s, sources)
    np.save(final_c, cids)
    np.save(final_p, paths)

    # Clean partial files
    for p in [chk_x, chk_y, chk_s, chk_c, chk_p]:
        if p.exists():
            try:
                os.remove(p)
            except Exception:
                pass

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Vision Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"  Total clips processed : {len(X)} in {total_time:.1f}s ({len(X)/total_time:.1f} clips/s)")
    print(f"  X shape               : {X.shape} ({X.nbytes / 1e6:.1f} MB)")
    print(f"  y shape               : {y.shape} (Fight={int(y.sum())}, NonFight={int((y==0).sum())})")
    print(f"  Per-dataset breakdown :")
    for src in sorted(set(sources)):
        m = sources == src
        print(f"    {src:<14}: Fight={int(y[m].sum()):>5}, NonFight={int((y[m]==0).sum()):>5} ({m.sum()} total)")
    print(f"  Saved to: {final_x.resolve()}\n")


if __name__ == "__main__":
    main()
