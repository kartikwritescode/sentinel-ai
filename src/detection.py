# yolo person detection

from ultralytics import YOLO
import config
import torch

class PersonDetector:
    """
    Wraps YOLO11n for real-time person detection + ByteTrack tracking.
    
    ByteTrack is built into Ultralytics -> calling model.track() instead of
    model.predict() activates it automatically. It assigns each person a
    persistent integer ID that stays the same across frames, even if they
    briefly leave the frame.
    """

    def __init__(self):
        # load the model once at startup -> expensive operation so better to not keep it inside loop
        self.model   = YOLO(config.YOLO_MODEL)   # load weights — the only thing YOLO() should receive
        self.conf    = config.YOLO_CONF_THRESH    # plain float, not a model path
        self.classes = config.YOLO_CLASSES        # plain list [0], not a model path
        self.device = 0 if torch.cuda.is_available() else "cpu"

    def detect_and_track(self,frame):
        """
        Args:
            frame: np.ndarray -> one BGR image from your video source
        
        Returns:
            list of dicts, one per detected person:
            [
                {
                    "track_id": 1,          # stable ID from ByteTrack
                    "bbox": [x1, y1, x2, y2],  # pixel coordinates
                    "confidence": 0.87,
                    "crop": np.ndarray      # the cropped person image for pose
                },
                ...
            ]
        """


        results = self.model.track(
            source=frame,
            tracker="bytetrack.yaml",
            classes=self.classes,
            conf=self.conf,
            persist=True,
            device=self.device,                 # GPU acceleration (CUDA) if available
            verbose=False
        )

        persons = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            has_keypoints = (result.keypoints is not None and result.keypoints.data is not None)

            for i, box in enumerate(result.boxes):
                track_id = int(box.id) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                keypoints_flat = None
                if has_keypoints and i < len(result.keypoints.data):
                    keypoints_flat = result.keypoints.data[i].cpu().numpy().flatten()

                persons.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'keypoints': keypoints_flat
                })
        return persons