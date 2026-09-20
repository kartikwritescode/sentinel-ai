# Sentinel AI — Real-Time CCTV Suspicious Behavior & Fight Detection System

A production-ready, GPU-accelerated computer vision pipeline that detects fighting, violence, and suspicious human interactions from live CCTV camera feeds, RTSP streams, and uploaded video files.

Built with **YOLO11-Pose (CUDA)**, **ByteTrack**, **MobileNetV2 Visual Features**, **Temporal Motion Delta ($\Delta x_t$)**, **PyTorch BiLSTM + Multi-Head Attention**, **FastAPI**, **SQLite**, and **OpenCV**.

---

## 📐 Pipeline Architecture

```text
[Video Source] ──► [YOLO11-Pose + ByteTrack (CUDA)] ──► [MobileNetV2 + Motion Delta] ──► [BiLSTM + Multi-Head Attention] ──► [Gated Alerting & HUD]
(Webcam/RTSP/MP4)     (60+ FPS Cadence & Tracking)       (Spatial Semantics + Motion Flux)    (Calibrated Threat Probability)   (Telegram + SQLite + 60+ FPS HUD)
```

1. **High-Speed Detection & Tracking**: YOLO11-Pose detects people and extracts 17 COCO body keypoints on CUDA GPU with ByteTrack tracking persistence. Alternate-frame inference cadence delivers **80+ FPS** real-time throughput.
2. **Dual Spatial & Motion Flux Modeling**:
   - **Spatial Semantics**: 1280-D deep visual embeddings via MobileNetV2.
   - **Temporal Motion Delta**: Computes backward frame difference $\Delta x_t = x_t - x_{t-1}$. Static scenes and background clutter yield $\Delta x_t = \mathbf{0}$, eliminating false alarms from static room textures.
3. **Temporal Classification**: 2-layer Bidirectional LSTM (`hidden_size=128`) with Multi-Head Temporal Self-Attention and max-pooling aggregation predicts violent interaction probability over rolling temporal windows.
4. **Production Surveillance Gating**:
   - **0-Person Gate**: If no humans are present, confidence is locked strictly to `0.0%` (`SYSTEM NORMAL`).
   - **Single-Person Gate**: Capped at $\le 10\%$ (`SYSTEM NORMAL (1 PERSON DETECTED)`). Fights require multi-person interaction.
   - **Debounced Alerting**: Multi-person hysteresis debouncer requires $\ge 2$ people and sustained high confidence ($\ge 0.65$ for 5 steps) to confirm an alert. Automatically resets when people separate or motion returns to normal.

---

## 📊 Current Model Evaluation & Benchmark Metrics

Evaluated on a held-out test split of **667 unseen video clips** across all three major surveillance benchmarks (**RLVS**, **RWF-2000**, and **SCVD**):

| Metric | Benchmark Result | Description |
|---|---|---|
| **Overall Accuracy** | **86.36%** | Correct predictions across all 667 test clips |
| **Recall / Sensitivity** | **90.26%** | High fight detection rate (detects 9 out of 10 violent events) |
| **Precision** | **86.40%** | Reliability of positive alert triggers |
| **Macro F1-Score** | **85.98%** | Balanced unweighted harmonic mean across classes |
| **Weighted F1-Score** | **86.30%** | Sample-weighted harmonic mean |
| **Specificity** | **81.18%** | True Negative Rate (normal activity correctly recognized) |
| **False Positive Rate (FPR)** | **18.82%** | Significantly suppressed via motion delta & person gating |
| **Calibrated Threshold** | **0.52** | Optimal validation-calibrated operating point |
| **Model Inference Latency** | **0.029 ms / window** | Measured on NVIDIA GeForce RTX 4060 Laptop GPU |
| **Model Throughput** | **34,063 FPS** | Raw batch inference throughput |
| **Real-Time Surveillance FPS** | **81.3 FPS** | End-to-end pipeline (Capture + Pose + Tracking + Vision + Display) |

### 🎯 Confusion Matrix (667 Test Clips)

| | Predicted NonFight | Predicted Fight |
|---|---|---|
| **Actual NonFight (380 clips)** | **TN = 233** | FP = 54 |
| **Actual Fight (287 clips)** | FN = 37 | **TP = 343** |

---

### 🌐 Cross-Dataset Generalization Breakdown

The model is trained across all **6,648 clips** from diverse environments (school hallways, streets, bars, indoor rooms, night surveillance). Generalization is verified per dataset:

| Dataset Scope | Test Samples | Accuracy | Macro F1 | Correct / Total |
|---|---|---|---|---|
| **RLVS** (Real Life Violence Situations) | 173 clips | **94.22%** | **94.21%** | 163 / 173 clips |
| **SCVD** (Smart City Violence Dataset) | 276 clips | **93.48%** | **92.54%** | 258 / 276 clips |
| **RWF-2000** (Real-World Surveillance CCTV) | 218 clips | **71.10%** | **70.88%** | 155 / 218 clips |
| **Overall Combined** | **667 clips** | **86.36%** | **85.98%** | **576 / 667 clips** |

---

## 🛡️ False Alarm Suppression & Production Behavior

| Camera State | Confidence Output | HUD Status Display | Alert State |
|---|---|---|---|
| **Empty Room (0 Persons)** | **0.0%** | `SYSTEM NORMAL (NO PERSONS DETECTED)` | Normal (Green) |
| **Single Person Sitting / Walking** | **$\le 8.8\%$** | `SYSTEM NORMAL (1 PERSON DETECTED)` | Normal (Green) |
| **Normal Multi-Person Interaction** | **$\le 15.0\%$** | `SYSTEM NORMAL \| People: N` | Normal (Green) |
| **Elevated Motion / Approaching** | **60.0% – 65.0%** | `WARNING: ELEVATED INTERACTION` | Warning (Yellow) |
| **Active Violent Brawl ($\ge 2$ People)** | **$\ge 65.0\%$ (Sustained)** | `ALERT: VIOLENT FIGHT DETECTED!` | Alarm (Red Banner + Telegram + SQLite) |

---

## ⚙️ Quickstart & Setup Guide

### 1. Prerequisites
* Python 3.10
* NVIDIA GPU with CUDA drivers (supports RTX series / GTX series)

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/kartikwritescode/sentinel-ai.git
cd sentinel-ai

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install / verify dependencies
pip install -r requirements.txt
```

---

## 🛠 Step-by-Step Workflow

### Step 1: Run Full System Self-Test Diagnostic
Verifies CUDA GPU, dataset files, feature arrays, model checkpoints, and inference speed:

```powershell
python test.py
```

### Step 2: Train Model on All Datasets (GPU Accelerated)
Trains `VisionBiLSTMClassifier` with dual spatial & motion-delta streams on all 6,648 clips (RLVS, RWF-2000, SCVD) using AMP FP16 on GPU:

```powershell
python scripts/train.py --model vision_bilstm --dataset all --epochs 35 --batch-size 64 --lr 1e-3
```

### Step 3: Evaluate Model Benchmark & Save JSON Report
Calculates accuracy, precision, recall, F1, confusion matrix, per-dataset breakdown, and hardware latency:

```powershell
python scripts/evaluate.py --model-type vision_bilstm
```

### Step 4: Run Live Real-Time Surveillance at 60+ FPS

* **Run on Webcam (`0`)**:
  ```powershell
  python app.py
  ```

* **Run on Any Video File or Evidence Clip**:
  ```powershell
  python app.py data/evidence_clips/event_20260916_101534.mp4
  ```

* **Run on RTSP Security Camera Stream**:
  ```powershell
  python app.py "rtsp://username:password@192.168.1.100:554/stream1"
  ```
  *(Press `q` in the video window to stop).*

---

## 🌐 FastAPI REST Backend

Start the FastAPI application server:

```powershell
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at: `http://localhost:8000/docs` (Swagger UI).

### API Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System health check and device info |
| `GET` | `/metrics` | Returns model evaluation metrics JSON report |
| `GET` | `/events` | Queries recent logged events from SQLite database |
| `GET` | `/video_feed?source=0` | Real-time MJPEG video stream with visual HUD overlay |
| `POST` | `/upload_video` | Processes uploaded video file and returns incident report JSON |

---

## 📁 Repository Structure

```text
cctv/
├── config.py             # Centralized system, threshold & model configuration
├── app.py                # 60+ FPS live desktop surveillance application & HUD
├── test.py               # 5-stage automated self-test and diagnostic script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── src/
│   ├── api.py            # FastAPI REST server & MJPEG streaming endpoints
│   ├── video_source.py   # Multi-threaded unified video source manager
│   ├── detection.py      # YOLO11-Pose CUDA detector & ByteTrack tracker
│   ├── features.py       # Kinematic motion & pose interaction feature engineering
│   ├── classifier.py     # Vision-BiLSTM + Motion Delta + Multi-Head Attention model
│   └── alerting.py       # Telegram alerts, debouncer & SQLite event logger
├── scripts/
│   ├── extract_vision_features.py # GPU batch feature extraction (MobileNetV2)
│   ├── extract_features.py        # Kinematic feature extraction
│   ├── train.py          # Multi-dataset training pipeline with AMP & Focal Loss
│   └── evaluate.py       # Model evaluation & metrics report generator
├── data/
│   ├── events.db         # SQLite incident database
│   ├── eval_report.json  # Comprehensive evaluation metrics JSON report
│   ├── evidence_clips/   # Auto-saved incident MP4 clips
│   ├── X_vision.npy      # 6,648 visual feature sequences (15x1280)
│   ├── y_vision.npy      # Labels array
│   └── sources_vision.npy# Dataset origin tags (RLVS, RWF-2000, SCVD)
└── models/
    ├── vision_bilstm.pt  # Production Vision-BiLSTM model checkpoint
    └── tier2_gru.pt      # Kinematic BiGRU model checkpoint
```