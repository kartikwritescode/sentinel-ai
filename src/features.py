# math/engineering that turns poses into meaningful numbers that capture behaviour

# Person A's right wrist velocity is 8.2 m/s, they are 0.3 body-widths apart from Person B, their bounding boxes overlap by 40%


import numpy as np
import cv2
from collections import defaultdict, deque
import config

class FeatureEngineer:
    """
    Maintains a rolling history of detections per track_id and computes
    a feature vector over the last FEATURE_WINDOW_FRAMES frames.
    
    This is a STATEFUL object -> it remembers what it saw in past frames.
    You create ONE instance and call update() on every frame.
    """

    def __init__(self, window_size = config.FEATURE_WINDOW_FRAMES):
        self.window_size = window_size
        # per person history : track_id -> deque of keypoint arrays
        self.pose_history = defaultdict(lambda:deque(maxlen=window_size))
        # per person hisotry : track id -> deque of bounding boxes
        self.bbox_history = defaultdict(lambda:deque(maxlen=window_size))
        # previous frame (for optical flow)
        self._prev_gray = None 

        # defaultdict(lambda: deque(maxlen=window_size)): A dictionary that auto-creates a deque when you access a new key. The maxlen on the deque means old frames automatically fall off the end -> you always have exactly the last N frames, no manual trimming.
        

    def update(self,frame,persons_with_poses):
        """
        Call this every frame. Updates internal history.
        
        Args:
            frame: current BGR frame (np.ndarray)
            persons_with_poses: list of dicts:
              [{"track_id": 1, "bbox": [...], "keypoints": np.ndarray or None}, ...]
        
        Returns:
            feature_vector: np.ndarray — the computed features for THIS frame window
                            Returns None if not enough history yet
        """

        # store history for each tracked person
        for person in persons_with_poses:
            tid = person['track_id']
            if person['keypoints'] is not None:
                self.pose_history[tid].append(person['keypoints'])
            self.bbox_history[tid].append(person['bbox'])

        # compute optical flow (global motion signal)
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        flow_magnitude = self._compute_optical_flow(gray)
        self._prev_gray = gray

        # Compute per person features
        person_features = []
        track_ids = list(self.pose_history.keys())

        for tid in track_ids:
            hist = list(self.pose_history[tid])
            bbox_hist = list(self.bbox_history[tid])

            if len(hist) < 2: # need atleast 2 frames for velocity
                continue
        wrist_velocity = self._wrist_velocity(hist)
        person_features.append(wrist_velocity)

        # compute pairwise features (between every pair of people)
        
        pair_features = []
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid_a, tid_b = track_ids[i], track_ids[j]
                if tid_a in self.bbox_history and tid_b in self.bbox_history:
                    dist   = self._person_distance(tid_a, tid_b)
                    overlap = self._bbox_overlap(tid_a, tid_b)
                    pair_features.extend([dist, overlap])
        
        # Assemble final feature vector
        # Use mean/max aggregation so the vector is always the same size regardless of how many people are in the frame
        features = []
        features.append(np.mean(person_features) if person_features else 0.0)
        features.append(np.max(person_features)  if person_features else 0.0)
        features.append(np.mean(pair_features)   if pair_features   else 0.0)
        features.append(np.max(pair_features)    if pair_features   else 0.0)
        features.append(flow_magnitude)
        features.append(len(track_ids))  # person count as a feature
        
        return np.array(features, dtype=np.float32)
    
    def _wrist_velocity(self, pose_history):
        """
        Approximate wrist speed from the last 2 frames.
        MediaPipe landmark 15 = left wrist, 16 = right wrist.
        Returns the maximum wrist speed (left or right).
        """
        if len(pose_history) < 2:
            return 0.0
        prev_kp = pose_history[-2]
        curr_kp = pose_history[-1]
        
        # Each keypoint is stored as [x, y, visibility, x, y, visibility, ...]
        # Landmark index 15 → array position 15*3 = 45
        # Landmark index 16 → array position 16*3 = 48
        
        left_wrist_prev  = prev_kp[45:47]   # x, y of left wrist
        left_wrist_curr  = curr_kp[45:47]
        right_wrist_prev = prev_kp[48:50]
        right_wrist_curr = curr_kp[48:50]
        
        left_speed  = np.linalg.norm(left_wrist_curr  - left_wrist_prev)
        right_speed = np.linalg.norm(right_wrist_curr - right_wrist_prev)
        
        return float(max(left_speed, right_speed))
    
    def _person_distance(self, tid_a, tid_b):
        """
        Euclidean distance between the centers of two people's bounding boxes.
        Normalized by the average person height (bbox height) so it's
        scale-invariant across different camera heights.
        """
        bbox_a = self.bbox_history[tid_a][-1]
        bbox_b = self.bbox_history[tid_b][-1]
        
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        
        pixel_dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
        
        avg_height = ((bbox_a[3] - bbox_a[1]) + (bbox_b[3] - bbox_b[1])) / 2
        if avg_height == 0:
            return 0.0
        
        return float(pixel_dist / avg_height)  # unit: "person heights"
    
    def _bbox_overlap(self, tid_a, tid_b):
        """
        Intersection-over-Union (IoU) between two bounding boxes.
        High IoU = people are physically overlapping/in contact.
        Range: 0.0 (no overlap) to 1.0 (perfect overlap).
        """
        a = self.bbox_history[tid_a][-1]
        b = self.bbox_history[tid_b][-1]
        
        inter_x1 = max(a[0], b[0])
        inter_y1 = max(a[1], b[1])
        inter_x2 = min(a[2], b[2])
        inter_y2 = min(a[3], b[3])
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union_area = area_a + area_b - inter_area
        
        return float(inter_area / union_area) if union_area > 0 else 0.0
    
    def _compute_optical_flow(self, gray_frame):
        """
        Dense optical flow using Farneback method (built into OpenCV).
        Returns the mean magnitude of motion vectors across the whole frame.
        High value = lots of fast movement = potential alarm signal.
        """
        if self._prev_gray is None:
            return 0.0
        
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray_frame,
            None,
            pyr_scale=0.5,   # pyramid scale
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))
    
    def reset(self):
        """Clear all history. Call between unrelated video clips."""
        self.pose_history.clear()
        self.bbox_history.clear()
        self._prev_gray = None





