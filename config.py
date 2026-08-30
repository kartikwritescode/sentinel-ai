SEQUENCE_LENGTH = 16
IMAGE_HEIGHT , IMAGE_WIDTH = 64,64
# CLASSES_LIST = ['NonViolence','Violence']
CLASSES_LIST = ['NonFight','Fight']


# yolo 
YOLO_MODEL       = 'yolo11n-pose.pt'    # GPU-accelerated pose detection model
YOLO_CONF_THRESH = 0.4             
YOLO_CLASSES     = [0]             

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE  = 0.5


# how many frames to look back when computing velocity 
FEATURE_WINDOW_FRAMES = 30  # 1 second if 30fps vid 



TIER2_MODEL_PATH  = 'models/tier2_gru.pt'
# 24-D Kinematic & Multi-Person Spatial Interaction Feature Vector
TIER2_INPUT_SIZE  = 24
TIER2_HIDDEN_SIZE = 128
TIER2_NUM_LAYERS  = 2

# GPU / Training settings 
TRAINING_DEVICE     = 'cuda'
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS     = 60
LEARNING_RATE       = 1e-3

# Alerting & Hysteresis settings
ALERT_CONF_HIGH          = 0.75   # Threshold to trigger alert state
ALERT_CONF_LOW           = 0.40   # Threshold to release alert state (hysteresis)
ALERT_SUSTAINED_FRAMES   = 15     # Must sustain high confidence for >= 15 frames (~0.5s at 30fps)
ALERT_COOLDOWN_SECONDS   = 15.0   # Minimum seconds between successive alert dispatches
PRE_EVENT_BUFFER_SECONDS = 4      # Seconds of footage to save BEFORE the alert
POST_EVENT_RECORD_SECONDS= 3      # Seconds AFTER alert trigger to keep recording

SQLITE_DB_PATH     = 'data/events.db'
EVIDENCE_CLIPS_DIR = 'data/evidence_clips'

# loaded from .env 
import os
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID', '')