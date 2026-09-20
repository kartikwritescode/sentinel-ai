# 34-D Scale-Normalized Kinematic, Pose-Geometric, and Temporal Dynamics Feature Engineering
import numpy as np
import cv2
from collections import defaultdict, deque
import torch
import config


class FeatureEngineer:
    """
    Stateful kinematic and spatial interaction feature extractor.
    Extracts a 34-dimensional feature vector per frame capturing:

      Scale-Normalized Joint Velocities (0-9):
        - Wrist, elbow, shoulder, leg, head velocities normalized by bbox height

      Dynamics (10-12):
        - Acceleration, kinetic energy (normalized)

      Multi-Person Interaction (13-19):
        - Pairwise proximity, IoU, approach rate, strike distances
        - Sentinel value -1.0 when < 2 persons (instead of 0.0)

      Motion & Count (20-23):
        - Person-masked optical flow, aspect ratio change, person count

      Joint Angle & Pose Geometry (24-27):
        - Elbow flexion angle, arm raise score, wrist-hip displacement, torso lean

      Temporal Dynamics (28-31):
        - Velocity variance, acceleration oscillation, energy spike ratio, head drop rate

      Body Shape (32-33):
        - Skeleton compactness change, limb extension asymmetry
    """
    PAIRWISE_SENTINEL = -1.0  # Value for pairwise features when < 2 persons

    def __init__(self, window_size=getattr(config, 'FEATURE_WINDOW_FRAMES', 30)):
        self.window_size = window_size
        self.pose_history = defaultdict(lambda: deque(maxlen=window_size))
        self.bbox_history = defaultdict(lambda: deque(maxlen=window_size))
        self.velocity_history = defaultdict(lambda: deque(maxlen=window_size))
        self.ke_history = defaultdict(lambda: deque(maxlen=window_size))
        self.head_y_history = defaultdict(lambda: deque(maxlen=window_size))
        self._prev_gray = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, frame, persons_with_poses):
        """
        Processes the current frame and detected person poses.
        Returns:
            np.ndarray (shape: (34,)) or None
        """
        # 1. Update history for tracked individuals
        current_tids = set()
        for person in persons_with_poses:
            tid = person['track_id']
            current_tids.add(tid)
            if person.get('keypoints') is not None and len(person['keypoints']) >= 51:
                self.pose_history[tid].append(person['keypoints'])
            self.bbox_history[tid].append(person['bbox'])

        # Clean up stale track IDs
        all_tids = list(self.bbox_history.keys())
        for tid in all_tids:
            if tid not in current_tids and len(self.bbox_history[tid]) == 0:
                self.pose_history.pop(tid, None)
                self.bbox_history.pop(tid, None)
                self.velocity_history.pop(tid, None)
                self.ke_history.pop(tid, None)
                self.head_y_history.pop(tid, None)

        # 2. Compute Person-Masked Optical Flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_mean, flow_peak = self._compute_person_masked_flow(gray, persons_with_poses)
        self._prev_gray = gray

        # 3. Compute Per-Person Kinematic + Pose-Geometric Features
        wrist_vels, elbow_vels, shoulder_vels, leg_vels, head_vels = [], [], [], [], []
        accels, kinetic_energies, aspect_ratio_changes = [], [], []
        elbow_angles, arm_raise_scores, wrist_hip_disps = [], [], []
        torso_leans, head_drop_rates = [], []
        compactness_changes, limb_asymmetries = [], []
        vel_variances, accel_oscillations, energy_spikes = [], [], []

        for tid in current_tids:
            p_hist = list(self.pose_history[tid])
            b_hist = list(self.bbox_history[tid])
            bbox_curr = b_hist[-1] if b_hist else [0, 0, 1, 1]
            bbox_h = max(bbox_curr[3] - bbox_curr[1], 1.0)

            if len(p_hist) >= 2:
                # Scale-normalized joint speeds & kinetic energy
                w_v, e_v, s_v, l_v, h_v, k_e = self._compute_joint_dynamics(
                    p_hist[-2], p_hist[-1], bbox_h
                )
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

                # Track KE history for energy spike ratio (robust clamping prevents 2M+ explosions)
                self.ke_history[tid].append(k_e)
                if len(self.ke_history[tid]) >= 3:
                    ke_hist = list(self.ke_history[tid])
                    mean_ke = max(float(np.mean(ke_hist[:-1])), 0.05)
                    spike = float(np.clip(ke_hist[-1] / mean_ke, 0.0, 50.0))
                    energy_spikes.append(spike)

                # Velocity variance (erratic motion detection)
                if len(self.velocity_history[tid]) >= 3:
                    vel_variances.append(float(np.std(list(self.velocity_history[tid]))))

                # Acceleration oscillation (sign changes in velocity differences)
                v_hist = list(self.velocity_history[tid])
                if len(v_hist) >= 4:
                    diffs = [v_hist[j+1] - v_hist[j] for j in range(len(v_hist)-1)]
                    sign_changes = sum(1 for j in range(len(diffs)-1) if diffs[j] * diffs[j+1] < 0)
                    accel_oscillations.append(sign_changes / max(len(diffs)-1, 1))

            # Pose geometry features (only need current keypoints)
            if len(p_hist) >= 1:
                kp = p_hist[-1]

                # Elbow flexion angles
                angle_l = self._compute_elbow_angle(kp, side='left')
                angle_r = self._compute_elbow_angle(kp, side='right')
                valid_angles = [a for a in [angle_l, angle_r] if a is not None]
                if valid_angles:
                    elbow_angles.append(min(valid_angles))  # min = most bent = punch

                # Arm raise score
                raise_score = self._compute_arm_raise_score(kp, bbox_h)
                if raise_score is not None:
                    arm_raise_scores.append(raise_score)

                # Wrist-hip displacement
                whd = self._compute_wrist_hip_displacement(kp, bbox_h)
                if whd is not None:
                    wrist_hip_disps.append(whd)

                # Torso lean angle
                lean = self._compute_torso_lean(kp)
                if lean is not None:
                    torso_leans.append(lean)

                # Skeleton compactness
                compact = self._compute_skeleton_compactness(kp, bbox_curr)
                if compact is not None:
                    compactness_changes.append(compact)

                # Limb extension asymmetry
                asym = self._compute_limb_asymmetry(kp, bbox_h)
                if asym is not None:
                    limb_asymmetries.append(asym)

                # Head drop rate (track head Y position)
                head_y = self._get_head_y(kp)
                if head_y is not None:
                    self.head_y_history[tid].append(head_y)
                    if len(self.head_y_history[tid]) >= 2:
                        hy = list(self.head_y_history[tid])
                        drop = (hy[-1] - hy[-2]) / bbox_h  # positive = head moving down
                        head_drop_rates.append(drop)

            if len(b_hist) >= 2:
                # Aspect ratio rate of change
                ar_curr = (b_hist[-1][3] - b_hist[-1][1]) / max(b_hist[-1][2] - b_hist[-1][0], 1)
                ar_prev = (b_hist[-2][3] - b_hist[-2][1]) / max(b_hist[-2][2] - b_hist[-2][0], 1)
                aspect_ratio_changes.append(abs(ar_curr - ar_prev))

        # 4. Multi-Person Interaction & Striking Features
        tid_list = list(current_tids)
        pair_dists, pair_ious, pair_approaches, strike_dists = [], [], [], []
        has_pairs = len(tid_list) >= 2

        for i in range(len(tid_list)):
            for j in range(i + 1, len(tid_list)):
                ta, tb = tid_list[i], tid_list[j]
                if ta in self.bbox_history and tb in self.bbox_history:
                    d_curr, iou = self._bbox_interaction(ta, tb)
                    pair_dists.append(d_curr)
                    pair_ious.append(iou)

                    if len(self.bbox_history[ta]) >= 2 and len(self.bbox_history[tb]) >= 2:
                        d_prev, _ = self._bbox_interaction_at_offset(ta, tb, -2)
                        approach_speed = max(0.0, d_prev - d_curr)
                        pair_approaches.append(approach_speed)

                    if len(self.pose_history[ta]) > 0 and len(self.pose_history[tb]) > 0:
                        s_dist = self._extremity_to_body_distance(
                            self.pose_history[ta][-1], self.pose_history[tb][-1],
                            self.bbox_history[ta][-1], self.bbox_history[tb][-1]
                        )
                        if s_dist is not None:
                            strike_dists.append(s_dist)

        # Use sentinel value for pairwise features when < 2 persons
        S = self.PAIRWISE_SENTINEL

        # 5. Assemble 34-D Feature Vector
        features = [
            # --- Scale-Normalized Velocity Features (0-9) ---
            float(np.mean(wrist_vels))    if wrist_vels    else 0.0,
            float(np.max(wrist_vels))     if wrist_vels    else 0.0,
            float(np.mean(elbow_vels))    if elbow_vels    else 0.0,
            float(np.max(elbow_vels))     if elbow_vels    else 0.0,
            float(np.mean(shoulder_vels)) if shoulder_vels else 0.0,
            float(np.max(shoulder_vels))  if shoulder_vels else 0.0,
            float(np.mean(leg_vels))      if leg_vels      else 0.0,
            float(np.max(leg_vels))       if leg_vels      else 0.0,
            float(np.mean(head_vels))     if head_vels     else 0.0,
            float(np.max(head_vels))      if head_vels     else 0.0,
            # --- Dynamics (10-12) ---
            float(np.max(accels))             if accels           else 0.0,
            float(np.mean(kinetic_energies))  if kinetic_energies else 0.0,
            float(np.max(kinetic_energies))   if kinetic_energies else 0.0,
            # --- Pairwise Interaction (13-19) — sentinel when < 2 persons ---
            float(np.min(pair_dists))      if has_pairs and pair_dists      else S,
            float(np.mean(pair_dists))     if has_pairs and pair_dists      else S,
            float(np.max(pair_ious))       if has_pairs and pair_ious       else S,
            float(np.mean(pair_ious))      if has_pairs and pair_ious       else S,
            float(np.max(pair_approaches)) if has_pairs and pair_approaches else S,
            float(np.min(strike_dists))    if has_pairs and strike_dists    else S,
            float(np.mean(strike_dists))   if has_pairs and strike_dists    else S,
            # --- Motion & Count (20-23) ---
            float(flow_mean),
            float(flow_peak),
            float(np.max(aspect_ratio_changes)) if aspect_ratio_changes else 0.0,
            float(len(current_tids)),
            # --- Joint Angle & Pose Geometry (24-27) ---
            float(np.min(elbow_angles))       if elbow_angles     else 1.0,
            float(np.max(arm_raise_scores))   if arm_raise_scores else 0.0,
            float(np.max(wrist_hip_disps))    if wrist_hip_disps  else 0.0,
            float(np.max(torso_leans))        if torso_leans      else 0.0,
            # --- Temporal Dynamics (28-31) ---
            float(np.max(vel_variances))      if vel_variances      else 0.0,
            float(np.max(accel_oscillations)) if accel_oscillations else 0.0,
            float(np.max(energy_spikes))      if energy_spikes      else 1.0,
            float(np.max(head_drop_rates))    if head_drop_rates    else 0.0,
            # --- Body Shape (32-33) ---
            float(np.max(compactness_changes)) if compactness_changes else 0.0,
            float(np.max(limb_asymmetries))    if limb_asymmetries    else 0.0,
        ]

        return np.array(features, dtype=np.float32)

    # ─── Joint Dynamics (scale-normalized) ──────────────────────────────

    def _compute_joint_dynamics(self, kp_prev, kp_curr, bbox_height):
        """
        Computes joint velocities normalized by bounding-box height.
        This makes features invariant to camera distance and resolution.

        COCO 17 Index Mapping:
          0: Nose   5,6: Shoulders   7,8: Elbows   9,10: Wrists
          11,12: Hips   13,14: Knees   15,16: Ankles
        """
        def joint_speed(idx):
            p = idx * 3
            if kp_prev[p+2] > 0.25 and kp_curr[p+2] > 0.25:
                dx = kp_curr[p] - kp_prev[p]
                dy = kp_curr[p+1] - kp_prev[p+1]
                return float(np.sqrt(dx*dx + dy*dy)) / bbox_height
            return 0.0

        w_v = max(joint_speed(9), joint_speed(10))
        e_v = max(joint_speed(7), joint_speed(8))
        s_v = max(joint_speed(5), joint_speed(6))
        l_v = max(joint_speed(13), joint_speed(14), joint_speed(15), joint_speed(16))
        h_v = joint_speed(0)

        # Scale-normalized kinetic energy
        ke = 0.0
        for j in range(17):
            v = joint_speed(j)
            ke += (v * v)

        return w_v, e_v, s_v, l_v, h_v, ke

    # ─── Joint Angle & Pose Geometry ────────────────────────────────────

    @staticmethod
    def _compute_elbow_angle(kp, side='left'):
        """
        Computes the elbow flexion angle (shoulder-elbow-wrist) normalized to [0, 1].
        0 = fully bent (punch), 1 = straight arm (relaxed).
        """
        if side == 'left':
            s_idx, e_idx, w_idx = 5, 7, 9
        else:
            s_idx, e_idx, w_idx = 6, 8, 10

        sp, ep, wp = s_idx * 3, e_idx * 3, w_idx * 3
        if kp[sp+2] < 0.25 or kp[ep+2] < 0.25 or kp[wp+2] < 0.25:
            return None

        # Vectors: shoulder→elbow and wrist→elbow
        se = np.array([kp[sp] - kp[ep], kp[sp+1] - kp[ep+1]])
        we = np.array([kp[wp] - kp[ep], kp[wp+1] - kp[ep+1]])

        dot = np.dot(se, we)
        norm_se = np.linalg.norm(se) + 1e-8
        norm_we = np.linalg.norm(we) + 1e-8
        cos_angle = np.clip(dot / (norm_se * norm_we), -1.0, 1.0)
        angle = np.arccos(cos_angle)  # radians [0, pi]

        return float(angle / np.pi)  # normalized [0, 1]

    @staticmethod
    def _compute_arm_raise_score(kp, bbox_height):
        """
        How far above shoulder the wrist is, normalized by bbox height.
        Positive = arm raised. Returns max across both arms.
        """
        scores = []
        for s_idx, w_idx in [(5, 9), (6, 10)]:
            sp, wp = s_idx * 3, w_idx * 3
            if kp[sp+2] > 0.25 and kp[wp+2] > 0.25:
                # In image coords, y increases downward, so shoulder_y > wrist_y means raised
                raise_val = (kp[sp+1] - kp[wp+1]) / bbox_height
                scores.append(raise_val)
        return max(scores) if scores else None

    @staticmethod
    def _compute_wrist_hip_displacement(kp, bbox_height):
        """
        Horizontal displacement of wrists from hips, normalized by bbox height.
        Large values indicate reaching/punching forward.
        """
        disps = []
        for w_idx, h_idx in [(9, 11), (10, 12)]:
            wp, hp = w_idx * 3, h_idx * 3
            if kp[wp+2] > 0.25 and kp[hp+2] > 0.25:
                dx = abs(kp[wp] - kp[hp])
                dy = abs(kp[wp+1] - kp[hp+1])
                d = np.sqrt(dx*dx + dy*dy) / bbox_height
                disps.append(float(d))
        return max(disps) if disps else None

    @staticmethod
    def _compute_torso_lean(kp):
        """
        Angle of the torso (hip_midpoint → shoulder_midpoint) from vertical.
        Normalized to [0, 1] where 0 = upright, 1 = horizontal.
        """
        # Need both shoulders and both hips
        if all(kp[i*3+2] > 0.25 for i in [5, 6, 11, 12]):
            sh_mid_x = (kp[5*3] + kp[6*3]) / 2.0
            sh_mid_y = (kp[5*3+1] + kp[6*3+1]) / 2.0
            hip_mid_x = (kp[11*3] + kp[12*3]) / 2.0
            hip_mid_y = (kp[11*3+1] + kp[12*3+1]) / 2.0

            dx = abs(sh_mid_x - hip_mid_x)
            dy = abs(sh_mid_y - hip_mid_y) + 1e-8
            lean_angle = np.arctan2(dx, dy)  # 0 = upright, pi/2 = horizontal
            return float(lean_angle / (np.pi / 2.0))  # normalized [0, 1]
        return None

    @staticmethod
    def _get_head_y(kp):
        """Returns nose Y coordinate if visible."""
        if kp[2] > 0.25:
            return float(kp[1])
        return None

    @staticmethod
    def _compute_skeleton_compactness(kp, bbox):
        """
        Ratio of the skeleton's bounding area to the person bounding box area.
        Low values indicate crouching / grappling / defensive posture.
        """
        visible_x, visible_y = [], []
        for j in range(17):
            p = j * 3
            if kp[p+2] > 0.25:
                visible_x.append(kp[p])
                visible_y.append(kp[p+1])

        if len(visible_x) < 4:
            return None

        skel_w = max(visible_x) - min(visible_x)
        skel_h = max(visible_y) - min(visible_y)
        skel_area = skel_w * skel_h

        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area <= 0:
            return None

        return float(skel_area / bbox_area)

    @staticmethod
    def _compute_limb_asymmetry(kp, bbox_height):
        """
        |left_arm_length - right_arm_length| / bbox_height.
        High asymmetry indicates one-sided striking (e.g., one arm punching).
        """
        def arm_length(shoulder_idx, wrist_idx):
            sp, wp = shoulder_idx * 3, wrist_idx * 3
            if kp[sp+2] > 0.25 and kp[wp+2] > 0.25:
                dx = kp[sp] - kp[wp]
                dy = kp[sp+1] - kp[wp+1]
                return np.sqrt(dx*dx + dy*dy)
            return None

        left_len = arm_length(5, 9)
        right_len = arm_length(6, 10)

        if left_len is not None and right_len is not None:
            return float(abs(left_len - right_len) / bbox_height)
        return None

    # ─── Bounding Box Interaction (unchanged) ───────────────────────────

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

        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        iou = (inter / union) if union > 0 else 0.0

        return float(dist), float(iou)

    # Strike Distance 

    def _extremity_to_body_distance(self, kp_a, kp_b, bbox_a, bbox_b):
        """Measures minimum normalized distance between extremities of A and vital zones of B."""
        avg_h = ((bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])) / 2.0
        if avg_h <= 0:
            return None

        extremity_indices = [9, 10, 15, 16]
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

    # Person-Masked Optical Flow  

    def _compute_person_masked_flow(self, gray_frame, persons):
        """
        Computes motion differencing ONLY within bounding boxes of detected persons.
        Ignores background motion (e.g., cars, trees, camera shake).
        """
        if self._prev_gray is None:
            self._prev_gray = gray_frame
            return 0.0, 0.0

        if not persons:
            return 0.0, 0.0

        h, w = gray_frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in persons:
            bbox = p.get('bbox', [0, 0, 0, 0])
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        diff = cv2.absdiff(gray_frame, self._prev_gray)
        person_motion = cv2.bitwise_and(diff, diff, mask=mask)

        active_pixels = cv2.countNonZero(mask)
        if active_pixels > 0:
            mean_motion = float(np.sum(person_motion) / active_pixels)
            peak_motion = float(np.max(person_motion))
        else:
            mean_motion, peak_motion = 0.0, 0.0

        return mean_motion, peak_motion

    # Reset 

    def reset(self):
        """Clears rolling history between clips or on stream reset."""
        self.pose_history.clear()
        self.bbox_history.clear()
        self.velocity_history.clear()
        self.ke_history.clear()
        self.head_y_history.clear()
        self._prev_gray = None
