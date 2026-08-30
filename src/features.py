# Advanced 24-D Kinematic and Multi-Person Spatial Interaction Feature Engineering
import numpy as np
import cv2
from collections import defaultdict, deque
import torch
import config


class FeatureEngineer:
    """
    Stateful kinematic and spatial interaction feature extractor.
    Extracts a 24-dimensional feature vector per frame capturing:
      - Individual joint velocities (wrists, elbows, shoulders, legs, head)
      - Upper-body acceleration / striking jerk
      - Kinetic energy across the skeleton
      - Bounding box aspect-ratio dynamics (fall / ground tackle detection)
      - Cross-person pairwise proximity & IoU overlap
      - Rapid approach rate (closing distance before strike)
      - Extremity-to-vital-zone distances (wrists/feet to head/torso)
      - Person-masked optical flow (immune to background motion)
      - Active tracked person count
    """
    def __init__(self, window_size=getattr(config, 'FEATURE_WINDOW_FRAMES', 30)):
        self.window_size = window_size
        self.pose_history = defaultdict(lambda: deque(maxlen=window_size))
        self.bbox_history = defaultdict(lambda: deque(maxlen=window_size))
        self.velocity_history = defaultdict(lambda: deque(maxlen=window_size))
        self._prev_gray = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, frame, persons_with_poses):
        """
        Processes the current frame and detected person poses.
        Returns:
            np.ndarray (shape: (24,)) or None
        """
        # 1. Update history for tracked individuals
        current_tids = set()
        for person in persons_with_poses:
            tid = person['track_id']
            current_tids.add(tid)
            if person.get('keypoints') is not None and len(person['keypoints']) >= 51:
                self.pose_history[tid].append(person['keypoints'])
            self.bbox_history[tid].append(person['bbox'])

        # Clean up stale track IDs that haven't been seen in window_size frames
        all_tids = list(self.bbox_history.keys())
        for tid in all_tids:
            if tid not in current_tids and len(self.bbox_history[tid]) == 0:
                self.pose_history.pop(tid, None)
                self.bbox_history.pop(tid, None)
                self.velocity_history.pop(tid, None)

        # 2. Compute Person-Masked Optical Flow (foreground only)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_mean, flow_peak = self._compute_person_masked_flow(gray, persons_with_poses)
        self._prev_gray = gray

        # 3. Compute Per-Person Kinematic Features
        wrist_vels, elbow_vels, shoulder_vels, leg_vels, head_vels = [], [], [], [], []
        accels, kinetic_energies, aspect_ratio_changes = [], [], []

        for tid in current_tids:
            p_hist = list(self.pose_history[tid])
            b_hist = list(self.bbox_history[tid])

            if len(p_hist) >= 2:
                # Joint speeds & kinetic energy
                w_v, e_v, s_v, l_v, h_v, k_e = self._compute_joint_dynamics(p_hist[-2], p_hist[-1])
                wrist_vels.append(w_v)
                elbow_vels.append(e_v)
                shoulder_vels.append(s_v)
                leg_vels.append(l_v)
                head_vels.append(h_v)
                kinetic_energies.append(k_e)

                # Track velocity for acceleration / jerk
                self.velocity_history[tid].append(w_v)
                if len(self.velocity_history[tid]) >= 2:
                    accel = abs(self.velocity_history[tid][-1] - self.velocity_history[tid][-2])
                    accels.append(accel)

            if len(b_hist) >= 2:
                # Aspect ratio rate of change: (h/w)_curr - (h/w)_prev
                ar_curr = (b_hist[-1][3] - b_hist[-1][1]) / max(b_hist[-1][2] - b_hist[-1][0], 1)
                ar_prev = (b_hist[-2][3] - b_hist[-2][1]) / max(b_hist[-2][2] - b_hist[-2][0], 1)
                aspect_ratio_changes.append(abs(ar_curr - ar_prev))

        # 4. Compute Multi-Person Interaction & Striking Features
        tid_list = list(current_tids)
        pair_dists, pair_ious, pair_approaches, strike_dists = [], [], [], []

        for i in range(len(tid_list)):
            for j in range(i + 1, len(tid_list)):
                ta, tb = tid_list[i], tid_list[j]
                if ta in self.bbox_history and tb in self.bbox_history:
                    # Normalized center distance & IoU
                    d_curr, iou = self._bbox_interaction(ta, tb)
                    pair_dists.append(d_curr)
                    pair_ious.append(iou)

                    # Approach velocity
                    if len(self.bbox_history[ta]) >= 2 and len(self.bbox_history[tb]) >= 2:
                        d_prev, _ = self._bbox_interaction_at_offset(ta, tb, -2)
                        approach_speed = max(0.0, d_prev - d_curr)
                        pair_approaches.append(approach_speed)

                    # Extremity-to-vital zone distance (Person A arms/legs to Person B torso/head)
                    if len(self.pose_history[ta]) > 0 and len(self.pose_history[tb]) > 0:
                        s_dist = self._extremity_to_body_distance(
                            self.pose_history[ta][-1], self.pose_history[tb][-1],
                            self.bbox_history[ta][-1], self.bbox_history[tb][-1]
                        )
                        if s_dist is not None:
                            strike_dists.append(s_dist)

        # 5. Assemble 24-D Feature Vector
        features = [
            float(np.mean(wrist_vels)) if wrist_vels else 0.0,
            float(np.max(wrist_vels))  if wrist_vels else 0.0,
            float(np.mean(elbow_vels)) if elbow_vels else 0.0,
            float(np.max(elbow_vels))  if elbow_vels else 0.0,
            float(np.mean(shoulder_vels)) if shoulder_vels else 0.0,
            float(np.max(shoulder_vels))  if shoulder_vels else 0.0,
            float(np.mean(leg_vels)) if leg_vels else 0.0,
            float(np.max(leg_vels))  if leg_vels else 0.0,
            float(np.mean(head_vels)) if head_vels else 0.0,
            float(np.max(head_vels))  if head_vels else 0.0,
            float(np.max(accels)) if accels else 0.0,
            float(np.mean(kinetic_energies)) if kinetic_energies else 0.0,
            float(np.max(kinetic_energies))  if kinetic_energies else 0.0,
            float(np.min(pair_dists)) if pair_dists else 0.0,
            float(np.mean(pair_dists)) if pair_dists else 0.0,
            float(np.max(pair_ious)) if pair_ious else 0.0,
            float(np.mean(pair_ious)) if pair_ious else 0.0,
            float(np.max(pair_approaches)) if pair_approaches else 0.0,
            float(np.min(strike_dists)) if strike_dists else 1.0,
            float(np.mean(strike_dists)) if strike_dists else 1.0,
            float(flow_mean),
            float(flow_peak),
            float(np.max(aspect_ratio_changes)) if aspect_ratio_changes else 0.0,
            float(len(current_tids))
        ]

        return np.array(features, dtype=np.float32)

    def _compute_joint_dynamics(self, kp_prev, kp_curr):
        """
        COCO 17 Index Mapping:
          0: Nose (Head)
          5, 6: Shoulders
          7, 8: Elbows
          9, 10: Wrists
          11, 12: Hips
          13, 14: Knees
          15, 16: Ankles
        """
        def joint_speed(idx):
            p = idx * 3
            if kp_prev[p+2] > 0.25 and kp_curr[p+2] > 0.25:
                dx = kp_curr[p] - kp_prev[p]
                dy = kp_curr[p+1] - kp_prev[p+1]
                return float(np.sqrt(dx*dx + dy*dy))
            return 0.0

        # Speeds
        w_v = max(joint_speed(9), joint_speed(10))
        e_v = max(joint_speed(7), joint_speed(8))
        s_v = max(joint_speed(5), joint_speed(6))
        l_v = max(joint_speed(13), joint_speed(14), joint_speed(15), joint_speed(16))
        h_v = joint_speed(0)

        # Kinetic energy estimate: sum of squared speeds across all visible joints
        ke = 0.0
        for j in range(17):
            v = joint_speed(j)
            ke += (v * v)

        return w_v, e_v, s_v, l_v, h_v, ke

    def _bbox_interaction(self, ta, tb):
        ba = self.bbox_history[ta][-1]
        bb = self.bbox_history[tb][-1]
        return self._calc_dist_and_iou(ba, bb)

    def _bbox_interaction_at_offset(self, ta, tb, offset):
        ba = self.bbox_history[ta][offset]
        bb = self.bbox_history[tb][offset]
        return self._calc_dist_and_iou(ba, bb)

    @staticmethod
    def _calc_dist_and_iou(a, b):
        cx_a, cy_a = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        cx_b, cy_b = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        h_avg = ((a[3] - a[1]) + (b[3] - b[1])) / 2.0
        h_avg = max(h_avg, 1.0)
        
        dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2) / h_avg

        # IoU
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        iou = (inter / union) if union > 0 else 0.0

        return float(dist), float(iou)

    def _extremity_to_body_distance(self, kp_a, kp_b, bbox_a, bbox_b):
        """Measures minimum normalized distance between extremities of A and vital zones of B."""
        avg_h = ((bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])) / 2.0
        if avg_h <= 0:
            return None

        # Extremity joints: Wrists (9, 10), Ankles (15, 16)
        extremity_indices = [9, 10, 15, 16]
        # Vital target zones: Head (0), Shoulders (5, 6), Torso/Hips (11, 12)
        target_indices = [0, 5, 6, 11, 12]

        min_d = float('inf')
        for e_idx in extremity_indices:
            ep = e_idx * 3
            if kp_a[ep+2] < 0.25:
                continue
            ex, ey = kp_a[ep], kp_a[ep+1]

            for t_idx in target_indices:
                tp = t_idx * 3
                if kp_b[tp+2] < 0.25:
                    continue
                tx, ty = kp_b[tp], kp_b[tp+1]
                d = np.sqrt((ex - tx)**2 + (ey - ty)**2) / avg_h
                if d < min_d:
                    min_d = d

        return min_d if min_d != float('inf') else 1.0

    def _compute_person_masked_flow(self, gray_frame, persons):
        """
        Computes motion differencing ONLY within bounding boxes of detected persons.
        Completely ignores background motion (e.g., cars, trees, camera shake).
        """
        if self._prev_gray is None:
            self._prev_gray = gray_frame
            return 0.0, 0.0

        if not persons:
            return 0.0, 0.0

        h, w = gray_frame.shape
        # Create lightweight binary mask for detected people
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in persons:
            bbox = p.get('bbox', [0, 0, 0, 0])
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        # Frame differencing on masked region
        diff = cv2.absdiff(gray_frame, self._prev_gray)
        person_motion = cv2.bitwise_and(diff, diff, mask=mask)

        # Normalize metrics
        active_pixels = cv2.countNonZero(mask)
        if active_pixels > 0:
            mean_motion = float(np.sum(person_motion) / active_pixels)
            peak_motion = float(np.max(person_motion))
        else:
            mean_motion, peak_motion = 0.0, 0.0

        return mean_motion, peak_motion

    def reset(self):
        """Clears rolling history between clips or on stream reset."""
        self.pose_history.clear()
        self.bbox_history.clear()
        self.velocity_history.clear()
        self._prev_gray = None






