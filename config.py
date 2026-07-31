SEQUENCE_LENGTH = 16
IMAGE_HEIGHT , IMAGE_WIDTH = 64,64
# CLASSES_LIST = ['NonViolence','Violence']
CLASSES_LIST = ['NonFight','Fight']


# yolo 
YOLO_MODEL       = 'yolo11n.pt'    
YOLO_CONF_THRESH = 0.4             
YOLO_CLASSES     = [0]             

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE  = 0.5


# how many frames to look back when computing velocity 
FEATURE_WINDOW_FRAMES = 30  # 1 second if 30fps vid 



TIER2_MODEL_PATH = 'models/tier2_gru.pt'
TIER2_INPUT_SIZE = 18              # number of engineered features (you'll tune this)
TIER2_HIDDEN_SIZE = 64
TIER2_NUM_LAYERS  = 2