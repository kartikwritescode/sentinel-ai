CLASSES_LIST = ['NonFight','Fight']

# YOLO Pose Detection
YOLO_MODEL       = 'yolo11n-pose.pt'    # GPU-accelerated pose detection model
YOLO_CONF_THRESH = 0.4             
YOLO_CLASSES     = [0]             

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE  = 0.5

# Feature Engineering
FEATURE_WINDOW_FRAMES = 30  # 1 second at 30fps

# Tier-1 Vision Classifier (MobileNetV2 + BiLSTM + Attention)
VISION_IMG_SIZE       = 128
VISION_FRAME_COUNT    = 15
VISION_EMBEDDING_DIM  = 1280
VISION_MODEL_PATH     = 'models/vision_bilstm.pt'
HYBRID_MODEL_PATH     = 'models/hybrid_model.pt'
MODEL_TYPE            = 'vision_bilstm'  # 'vision_bilstm', 'pose_gru', 'hybrid'

# Tier-2 Kinematic Classifier
TIER2_MODEL_PATH  = 'models/tier2_gru.pt'
# 34-D Kinematic + Joint Angle + Temporal Dynamics + Body Shape Feature Vector
TIER2_INPUT_SIZE  = 34
TIER2_HIDDEN_SIZE = 128
TIER2_NUM_LAYERS  = 2

# GPU / Training settings 
TRAINING_DEVICE     = 'cuda'
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS     = 120
LEARNING_RATE       = 1e-3

# High-Sensitivity Real-Time Detection Settings (Multi-Dataset Production Calibrated)
ALERT_CONF_HIGH          = 0.65   # Threshold to trigger alert state
ALERT_CONF_LOW           = 0.35   # Threshold to release alert state (hysteresis release)
ALERT_SUSTAINED_FRAMES   = 5      # Must sustain confidence for >= 5 steps (~0.25s) to confirm fight
ALERT_COOLDOWN_SECONDS   = 10.0   # Minimum seconds between successive alert dispatches
PRE_EVENT_BUFFER_SECONDS = 4      # Seconds of footage to save BEFORE the alert
POST_EVENT_RECORD_SECONDS= 3      # Seconds AFTER alert trigger to keep recording

# Temporal Horizon Settings
INFERENCE_BUFFER_FRAMES  = 30     # Total rolling frame horizon (~1.0s to 1.5s real action)
INFERENCE_EVAL_INTERVAL  = 3      # Run inference evaluation every 3 frames for smooth 60-80 FPS
ENABLE_ROI_INSPECTION    = False  # Full-scene frame matching SCVD dataset resolution

SQLITE_DB_PATH     = 'data/events.db'
EVIDENCE_CLIPS_DIR = 'data/evidence_clips'

# loaded from .env 
import os
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID', '')