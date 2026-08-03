import mediapipe as mp
import numpy as np
import config

class PoseEstimator:
    """
    Runs MediaPipe Pose on a person's cropped image.
    
    MediaPipe gives you 33 landmarks. Each landmark has:
    - x, y: normalized coordinates (0.0 to 1.0 relative to crop size)
    - z: depth estimate (approximate, don't over-rely on it)
    - visibility: confidence that this joint is visible (0.0 to 1.0)
    
    We return a flat numpy array of shape (33 * 3,) = 99 values,
    or None if no pose is detected in the crop.
    """

    def __init__(self):
        mp_pose = mp.solutions.pose

        # model_complexity = 0 -> fastest , good enough for CCTV resolution

        self.pose = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence = config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence = config.POSE_MIN_TRACKING_CONFIDENCE,
            static_image_mode = False # tracking mode across frames -> 5x faster!
        )

    def extract_keypoints(self,bgr_crop):
        """
        Args:
            bgr_crop: np.ndarray — cropped person region from detection
        
        Returns:
            np.ndarray of shape (99,) — 33 landmarks × (x, y, visibility)
            or None if pose not detected
        """
        if bgr_crop.size ==0 :
            return None

        # MediaPipe expects RGB , OpenCV gives us BGR
        rgb_crop = bgr_crop[...,::-1]

        results = self.pose.process(rgb_crop)

        if not results.pose_landmarks:
            return None

        keypoints = []

        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x,landmark.y,landmark.visibility])

        return np.array(keypoints, dtype = np.float32)

    def close(self):
        self.pose.close()


    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
    
