"""
FastAPI Backend Server for Smart CCTV Surveillance System.

Endpoints:
  GET  /              - System health and server info
  GET  /metrics       - Returns model accuracy, precision, recall, F1, and latency report
  GET  /events        - Queries recent suspicious activity events logged in SQLite database
  GET  /video_feed    - Real-time MJPEG video stream (supports webcam, IP camera URL, or video file)
  POST /upload_video  - Processes uploaded video file, detects suspicious events, and returns incident report
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import json
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from src.video_source import VideoSource
from src.detection    import PersonDetector
from src.features     import FeatureEngineer
from src.classifier   import Tier2Inferencer
from src.alerting     import EventLogger, EvidenceClipWriter, AlertDebouncer, send_telegram_alert
from app import draw_visual_overlays

app = FastAPI(
    title="Smart CCTV Suspicious Behavior Detection API",
    description="Real-time GPU-accelerated violence & suspicious activity detection REST & Streaming API",
    version="1.0.0"
)

# Enable CORS for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "Smart CCTV Suspicious Behavior Detection System",
        "version": "1.0.0",
        "yolo_model": config.YOLO_MODEL,
        "device": config.TRAINING_DEVICE
    }


@app.get("/metrics")
def get_metrics():
    """Returns evaluation metrics report (accuracy, precision, recall, F1, latency)."""
    report_path = Path("data/eval_report.json")
    if not report_path.exists():
        # Run evaluation if report doesn't exist yet
        try:
            from scripts.evaluate import evaluate_model
            evaluate_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate metrics report: {str(e)}")

    with open(report_path, "r") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.get("/events")
def get_events(limit: int = Query(50, ge=1, le=500)):
    """Returns recent suspicious activity events logged in SQLite database."""
    logger = EventLogger()
    events = logger.get_recent_events(limit=limit)
    logger.close()

    event_list = []
    for ev in events:
        event_list.append({
            "id": ev[0],
            "timestamp": ev[1],
            "confidence": round(ev[2], 4),
            "clip_path": ev[3],
            "source_id": ev[4],
            "person_count": ev[5]
        })

    return {"count": len(event_list), "events": event_list}


def generate_mjpeg_stream(video_source_arg: str):
    """Generator function yielding JPEG frames for MJPEG HTTP streaming."""
    # Convert digit string to int if webcam index
    source_val = int(video_source_arg) if video_source_arg.isdigit() else video_source_arg

    detector   = PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = Tier2Inferencer()
    debouncer  = AlertDebouncer()
    logger     = EventLogger()
    clip_writer = EvidenceClipWriter()

    latest_confidence = None
    is_alerting = False

    with VideoSource(source_val) as source:
        fps = source.get_fps()
        clip_writer.fps = fps

        for frame in source:
            # 1. Detect people + pose on GPU in a single pass
            persons = detector.detect_and_track(frame)

            # 2. Compute motion & interaction features
            feature_vec = engineer.update(frame, persons)

            # 3. Maintain pre-event clip buffer
            clip_writer.push_frame(frame)

            if feature_vec is not None:
                conf = inferencer.push_features(feature_vec)
                if conf is not None:
                    latest_confidence = conf
                    is_alerting = debouncer.update(latest_confidence)

                    if is_alerting:
                        clip_path = clip_writer.trigger_save()
                        timestamp = logger.log_event(
                            confidence=latest_confidence,
                            clip_path=clip_path,
                            source_id=str(source_val),
                            person_count=len(persons)
                        )
                        send_telegram_alert(
                            f"⚠️ Suspicious activity detected!\n"
                            f"Time: {timestamp}\n"
                            f"Confidence: {latest_confidence:.1%}\n"
                            f"Persons: {len(persons)}"
                        )

            # 4. Render visual HUD overlays
            draw_visual_overlays(frame, persons, latest_confidence, is_alerting)

            # 5. Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
def stream_video_feed(source: str = Query("0", description="Video source: '0' for webcam, IP camera URL (http/rtsp), or file path")):
    """
    Real-time MJPEG Video Streaming Endpoint.
    Can be opened directly in any browser or embedded in <img src="/video_feed?source=0">
    """
    return StreamingResponse(
        generate_mjpeg_stream(source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/upload_video")
async def upload_and_process_video(file: UploadFile = File(...)):
    """
    Processes an uploaded video file (.mp4, .avi, .mov), detects suspicious events,
    logs incidents, and returns an incident report JSON.
    """
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Supported: {allowed_exts}")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"upload_{int(time.time())}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detector   = PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = Tier2Inferencer()
    debouncer  = AlertDebouncer()
    logger     = EventLogger()
    clip_writer = EvidenceClipWriter()

    incidents = []
    frame_count = 0
    latest_confidence = None

    with VideoSource(str(temp_path)) as source:
        fps = source.get_fps()
        clip_writer.fps = fps

        for frame in source:
            frame_count += 1
            persons = detector.detect_and_track(frame)
            feature_vec = engineer.update(frame, persons)
            clip_writer.push_frame(frame)

            if feature_vec is not None:
                conf = inferencer.push_features(feature_vec)
                if conf is not None:
                    latest_confidence = conf
                    is_alerting = debouncer.update(latest_confidence)

                    if is_alerting:
                        clip_path = clip_writer.trigger_save()
                        timestamp = logger.log_event(
                            confidence=latest_confidence,
                            clip_path=clip_path,
                            source_id=file.filename,
                            person_count=len(persons)
                        )
                        incidents.append({
                            "frame_index": frame_count,
                            "timestamp": timestamp,
                            "confidence": round(latest_confidence, 4),
                            "evidence_clip": clip_path,
                            "person_count": len(persons)
                        })

    return {
        "status": "success",
        "filename": file.filename,
        "total_frames_processed": frame_count,
        "incidents_detected": len(incidents),
        "incidents": incidents
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
