# Sentinel AI — Real-Time CCTV Suspicious Behavior & Fight Detection System

A production-ready, GPU-accelerated computer vision pipeline that detects fighting, violence, and suspicious human interactions from live CCTV camera feeds, RTSP streams, and uploaded video files.

Built with **YOLO11-Pose (CUDA)**, **ByteTrack**, **PyTorch BiGRU**, **FastAPI**, **SQLite**, and **OpenCV**.

---

## 📐 Pipeline Architecture

```text
[Video Source] ──► [YOLO11-Pose + ByteTrack (CUDA)] ──► [Feature Engineering] ──► [BiGRU Classifier (CUDA)] ──► [Alerting & HUD]
(Webcam / RTSP / MP4)     (1 Pass: Detect + Track + Pose)   (Velocity, Proximity, Flow)    (Suspicion Probability)    (Telegram + SQLite + OpenCV HUD)
```

1. **Detection & Tracking**: YOLO11-Pose detects people and extracts 17 COCO body keypoints on CUDA GPU in a single pass while ByteTrack assigns persistent track IDs across frames.
2. **Feature Engineering**: Computes rolling motion metrics per frame:
   - Maximum & mean wrist velocity (sudden arm movements)
   - Inter-person distance & bounding box IoU overlap (physical proximity & contact)
   - PyTorch GPU tensor motion differencing (global optical flow signal)
3. **Temporal Classification**: A 2-layer Bidirectional GRU (`hidden_size=128`) predicts suspicious activity probability over a sliding 30-frame window.
4. **Debounced Alerting**: Requires $N$ consecutive positive windows to fire, reducing single-frame false alarms. Logs events to SQLite (`data/events.db`), auto-saves 5-second evidence clips, and dispatches Telegram notifications.

---

## 📊 Model Evaluation & Metrics

Evaluate the trained model on held-out test splits using `scripts/evaluate.py`:

| Metric | Result | Description |
|---|---|---|
| **Accuracy** | **79.30%** | Overall correct predictions |
| **Precision** | **77.21%** | Positive predictive value |
| **Recall / Sensitivity** | **83.00%** | Proportion of actual fights detected |
| **Specificity** | **75.62%** | True Negative Rate (NonFight recognition) |
| **F1-Score** | **80.00%** | Harmonic mean of Precision & Recall |
| **False Positive Rate (FPR)** | **24.38%** | Rate of false alarms on normal activity |
| **GPU Inference Latency** | **0.027 ms** | Per feature window on RTX 4060 CUDA GPU |
| **GPU Throughput** | **36,425 FPS** | Batch inference capability |

### 🌐 Cross-Dataset Generalization

To ensure the model generalizes across different camera setups and lighting conditions, accuracy is benchmarked separately across datasets:

| Dataset | Test Accuracy | Sample Count |
|---|---|---|
| **RLVS** (Real Life Violence Situations) | **80.98%** | 205 clips |
| **RWF-2000** (Surveillance Footage) | **77.55%** | 196 clips |

---

## ⚙️ Quickstart & Setup Guide

### 1. Prerequisites
* Python 3.10
* NVIDIA GPU with CUDA drivers (optional, falls back to CPU)

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/kartikwritescode/sentinel-ai.git
cd sentinel-ai

# Install dependencies (CPU PyTorch)
py -3.10 -m pip install -r requirements.txt

# (Optional) Upgrade to CUDA PyTorch for GPU acceleration
py -3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 🛠 Step-by-Step Workflow

### Step 1: Extract Features (100% GPU Accelerated)
Processes video clips from `data/violence_fight_detection_dataset/`, extracts keypoints and motion features, and saves `data/X_tier2.npy`, `data/y_tier2.npy`, and `data/sources.npy`.

```powershell
py -3.10 scripts/extract_features.py
```

### Step 2: Train the Tier 2 BiGRU Model
Trains the PyTorch BiGRU on GPU using mini-batch gradient descent and stratified 80/10/10 data splitting. Saves checkpoint to `models/tier2_gru.pt`.

```powershell
py -3.10 scripts/train.py
```

### Step 3: Evaluate Model Metrics
Calculates accuracy, precision, recall, F1, confusion matrix, and latency report. Saves output to `data/eval_report.json`.

```powershell
py -3.10 scripts/evaluate.py
```

### Step 4: Run Live Real-Time Surveillance Window

* **Run on Laptop Webcam (`0`)**:
  ```powershell
  py -3.10 app.py
  ```

* **Run on Video File or IP Camera Stream**:
  ```powershell
  py -3.10 app.py "data/violence_fight_detection_dataset/RLVS/train/Fight/file_002001.mp4"
  ```
  *(Press `q` in the video window to stop).*

---

## 🌐 FastAPI REST Backend

Start the FastAPI application server:

```powershell
py -3.10 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at: `http://localhost:8000/docs` (Swagger UI).

### API Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System health check and device info |
| `GET` | `/metrics` | Returns model evaluation metrics JSON report |
| `GET` | `/events` | Queries recent logged events from SQLite database |
| `GET` | `/video_feed?source=0` | Real-time MJPEG video stream (supports webcam `0`, RTSP, or file path) |
| `POST` | `/upload_video` | Processes uploaded video file and returns incident report JSON |

---

## 🐳 Docker Deployment Guide

Containerize the application with Docker for easy deployment on Linux servers or cloud platforms.

### 1. Build the Docker Image

```bash
docker build -t sentinel-cctv:latest .
```

### 2. Run the Container

```bash
docker run -d \
  --name sentinel-cctv \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  sentinel-cctv:latest
```

The FastAPI server will be live at `http://localhost:8000`.

---

## 📁 Repository Structure

```text
cctv/
├── config.py             # Centralized system & model configuration
├── app.py                # Live desktop application entry point & visual HUD
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── .dockerignore         # Docker build exclusions
├── src/
│   ├── api.py            # FastAPI REST server & MJPEG streaming endpoints
│   ├── video_source.py   # Unified OpenCV video source manager
│   ├── detection.py      # YOLO11-Pose CUDA detector & ByteTrack tracker
│   ├── features.py       # Motion & pose interaction feature engineering
│   ├── classifier.py     # PyTorch BiGRU model & GPU trainer
│   └── alerting.py       # Telegram alerts & SQLite event logger
├── scripts/
│   ├── extract_features.py # Batch GPU feature extraction script
│   ├── train.py          # PyTorch BiGRU model training script
│   └── evaluate.py       # Model evaluation & metrics report generator
├── data/
│   ├── events.db         # SQLite incident database
│   ├── eval_report.json  # Generated evaluation metrics report
│   ├── evidence_clips/   # Auto-saved incident MP4 clips
│   ├── X_tier2.npy       # Extracted feature array
│   └── y_tier2.npy       # Labels array
└── models/
    └── tier2_gru.pt      # Trained PyTorch model checkpoint
```