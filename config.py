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
# must match the number of values FeatureEngineer.update() returns.
# Current features: mean_wrist_vel, max_wrist_vel, mean_pair_dist/iou, max_pair_dist/iou, optical_flow_magnitude, person_count = 6 total.
TIER2_INPUT_SIZE  = 6
TIER2_HIDDEN_SIZE = 128   # bumped from 64 — RTX 4060 handles this easily
TIER2_NUM_LAYERS  = 2

# GPU / Training settings 
# torch.device auto-selects CUDA if available, falls back to CPU
TRAINING_DEVICE     = 'cuda'   # force CUDA — will error loudly if no GPU found
TRAINING_BATCH_SIZE = 64       # mini-batch size — GPU needs batches to be fast
TRAINING_EPOCHS     = 80       # more epochs since GPU is fast enough


# alerting

ALERT_DEBOUNCE_WINDOWS = 3         # need 3 consecutive suspicious windows to fire
ALERT_CONFIDENCE_THRESH = 0.7     # minimum model confidence to even count a window
PRE_EVENT_BUFFER_SECONDS = 5      # seconds of footage to save BEFORE the alert
POST_EVENT_RECORD_SECONDS = 3     # seconds AFTER alert trigger to keep recording

SQLITE_DB_PATH = 'data/events.db'
EVIDENCE_CLIPS_DIR = 'data/evidence_clips'

# loaded from .env 
import os
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID', '')