"""
FastAPI Backend Server for Smart CCTV Surveillance System (Sentinel AI).

Endpoints:
  GET  /              - Live Web Dashboard & System Status
  GET  /health        - System health and device info
  GET  /metrics       - Returns model accuracy, precision, recall, F1, and latency report
  GET  /events        - Queries recent suspicious activity events logged in SQLite database
  GET  /video_feed    - Real-time MJPEG video stream (supports webcam, IP camera URL, or video file)
  POST /upload_video  - Processes uploaded video file, detects suspicious events, and returns incident report
  POST /settings      - Dynamically updates alert thresholds and sensitivity
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

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import config
from src.video_source import VideoSource
from src.detection    import PersonDetector
from src.features     import FeatureEngineer
from src.classifier   import Tier2Inferencer
from src.alerting     import EventLogger, EvidenceClipWriter, AlertDebouncer, send_telegram_alert, format_alert_message
from app import draw_visual_overlays

# Global Model Singletons
PIPELINE_MODELS = {
    "detector": None,
    "inferencer": None,
    "logger": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload GPU models once
    print("[API Lifespan] Initializing GPU model singletons...")
    PIPELINE_MODELS["detector"] = PersonDetector()
    PIPELINE_MODELS["inferencer"] = Tier2Inferencer()
    PIPELINE_MODELS["logger"] = EventLogger()
    print("[API Lifespan] Models loaded and ready.")
    yield
    # Shutdown
    print("[API Lifespan] Shutting down...")

app = FastAPI(
    title="Sentinel AI — Smart CCTV Suspicious Activity Detection API",
    description="Real-time GPU-accelerated violence & suspicious activity detection REST & Streaming API",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Interactive Live Web Surveillance Dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentinel AI — Live Surveillance Monitor</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            body { background-color: #0f172a; color: #f8fafc; padding: 20px; }
            header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid #334155; margin-bottom: 20px; }
            h1 { font-size: 1.5rem; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
            .badge { background: #0284c7; font-size: 0.75rem; padding: 4px 8px; border-radius: 9999px; }
            .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            .card { background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
            .video-container { position: relative; width: 100%; border-radius: 8px; overflow: hidden; background: #000; }
            .video-feed { width: 100%; height: auto; display: block; }
            .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; color: #94a3b8; }
            .event-list { max-height: 480px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
            .event-item { background: #0f172a; padding: 10px; border-radius: 6px; border-left: 4px solid #ef4444; font-size: 0.85rem; }
            .event-header { display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 4px; }
            .confidence { color: #f87171; }
            .controls { margin-top: 15px; display: flex; gap: 10px; }
            input, button { padding: 8px 14px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #fff; }
            button { background: #0284c7; cursor: pointer; font-weight: bold; border: none; }
            button:hover { background: #0369a1; }
        </style>
    </head>
    <body>
        <header>
            <h1>🛡️ SENTINEL AI <span class="badge">PROD v2.0</span></h1>
            <div>Status: <span style="color: #4ade80;">● ONLINE</span></div>
        </header>
        <div class="grid">
            <div class="card">
                <div class="card-title">Live Video Feed</div>
                <div class="video-container">
                    <img id="stream" class="video-feed" src="/video_feed?source=0" alt="CCTV Stream">
                </div>
                <div class="controls">
                    <input type="text" id="sourceInput" placeholder="Video source (0, RTSP, or video file)" value="0">
                    <button onclick="changeSource()">Switch Camera Source</button>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Recent Security Events</div>
                <div class="event-list" id="eventsList">
                    <div style="color: #64748b; font-size: 0.85rem;">Loading events...</div>
                </div>
            </div>
        </div>
        <script>
            function changeSource() {
                const src = document.getElementById('sourceInput').value;
                document.getElementById('stream').src = '/video_feed?source=' + encodeURIComponent(src);
            }
            async function fetchEvents() {
                try {
                    const res = await fetch('/events?limit=15');
                    const data = await res.json();
                    const container = document.getElementById('eventsList');
                    if (data.events.length === 0) {
                        container.innerHTML = '<div style="color: #64748b; font-size: 0.85rem;">No security events detected yet.</div>';
                        return;
                    }
                    container.innerHTML = data.events.map(ev => `
                        <div class="event-item">
                            <div class="event-header">
                                <span>Event #${ev.id}</span>
                                <span class="confidence">${(ev.confidence * 100).toFixed(1)}% Threat</span>
                            </div>
                            <div>Source: ${ev.source_id} | Persons: ${ev.person_count}</div>
                            <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">${ev.timestamp}</div>
                        </div>
                    `).join('');
                } catch (e) {
                    console.error("Error fetching events:", e);
                }
            }
            setInterval(fetchEvents, 3000);
            fetchEvents();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "Sentinel AI Suspicious Behavior Detection System",
        "version": "2.0.0",
        "yolo_model": config.YOLO_MODEL,
        "device": config.TRAINING_DEVICE
    }


@app.get("/metrics")
def get_metrics():
    """Returns evaluation metrics report (accuracy, precision, recall, F1, latency)."""
    report_path = Path("data/eval_report.json")
    if not report_path.exists():
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
    logger = PIPELINE_MODELS["logger"] or EventLogger()
    events = logger.get_recent_events(limit=limit)

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
    """Generator yielding optimized JPEG frames for real-time MJPEG HTTP streaming."""
    source_val = int(video_source_arg) if video_source_arg.isdigit() else video_source_arg

    detector   = PIPELINE_MODELS["detector"] or PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = PIPELINE_MODELS["inferencer"] or Tier2Inferencer()
    debouncer  = AlertDebouncer()
    logger     = PIPELINE_MODELS["logger"] or EventLogger()
    clip_writer = EvidenceClipWriter()

    latest_confidence = None
    frame_times = []

    with VideoSource(source_val) as source:
        fps = source.get_fps()
        clip_writer.fps = fps

        for frame in source:
            t0 = time.time()
            persons = detector.detect_and_track(frame)
            feature_vec = engineer.update(frame, persons)
            clip_writer.push_frame(frame)

            new_alert = False
            if feature_vec is not None:
                conf = inferencer.push_features(feature_vec)
                if conf is not None:
                    latest_confidence = conf
                    new_alert = debouncer.update(latest_confidence)

                    if new_alert:
                        clip_path = clip_writer.trigger_save()
                        timestamp = logger.log_event(
                            confidence=latest_confidence,
                            clip_path=clip_path,
                            source_id=str(source_val),
                            person_count=len(persons)
                        )
                        formatted_msg = format_alert_message(
                            timestamp=timestamp,
                            confidence=latest_confidence,
                            source_id=str(source_val),
                            person_count=len(persons)
                        )

                        def make_send_callback(msg):
                            def _callback(saved_video_path):
                                send_telegram_alert(message=msg, video_path=saved_video_path, async_mode=True)
                            return _callback

                        clip_writer.on_clip_complete = make_send_callback(formatted_msg)
                        send_telegram_alert(message=formatted_msg, async_mode=True)

            is_alarm_active = debouncer.is_alarm_active()

            t1 = time.time()
            frame_times.append(t1 - t0)
            if len(frame_times) > 15:
                frame_times.pop(0)
            fps_display = 1.0 / np.mean(frame_times) if frame_times else 30.0

            draw_visual_overlays(frame, persons, latest_confidence, is_alarm_active, fps_display)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
def stream_video_feed(source: str = Query("0", description="Video source: '0' for webcam, IP camera URL (http/rtsp), or file path")):
    """Real-time MJPEG Video Streaming Endpoint."""
    return StreamingResponse(
        generate_mjpeg_stream(source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/upload_video")
async def upload_and_process_video(file: UploadFile = File(...)):
    """Processes uploaded video file and returns incident report JSON."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Supported: {allowed_exts}")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"upload_{int(time.time())}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detector   = PIPELINE_MODELS["detector"] or PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = PIPELINE_MODELS["inferencer"] or Tier2Inferencer()
    debouncer  = AlertDebouncer()
    logger     = PIPELINE_MODELS["logger"] or EventLogger()
    clip_writer = EvidenceClipWriter()

    incidents = []
    frame_count = 0
    latest_confidence = None

    with VideoSource(str(temp_path), threaded=False) as source:
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
                    new_alert = debouncer.update(latest_confidence)

                    if new_alert:
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
