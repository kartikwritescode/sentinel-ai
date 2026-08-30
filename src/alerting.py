# Telegram notifications + SQLite logging + Asynchronous Alert Pipeline
import sqlite3
import os
import time
import datetime
import requests
import cv2
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import config

# Global asynchronous background task executor for non-blocking alerting & network calls
_ALERT_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="AlertWorker")


class EventLogger:
    """
    Thread-safe event logging to SQLite database.
    """
    def __init__(self, db_path=config.SQLITE_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_table(self):
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    confidence   REAL    NOT NULL,
                    clip_path    TEXT,
                    source_id    TEXT,
                    person_count INTEGER
                )
                """)
                conn.commit()
            finally:
                conn.close()

    def log_event(self, confidence, clip_path=None, source_id='unknown', person_count=0):
        timestamp = datetime.datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO events(timestamp, confidence, clip_path, source_id, person_count) "
                    "VALUES(?,?,?,?,?)",
                    (timestamp, float(confidence), clip_path, str(source_id), int(person_count))
                )
                conn.commit()
            finally:
                conn.close()
        return timestamp

    def get_recent_events(self, limit=50):
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                )
                rows = cursor.fetchall()
            finally:
                conn.close()
        return rows

    def close(self):
        pass


class EvidenceClipWriter:
    """
    Non-blocking rolling buffer and async video clip writer.
    Maintains pre-event footage in RAM and writes post-event footage
    asynchronously to prevent blocking the computer vision loop.
    """
    def __init__(self, fps=30, on_clip_complete=None):
        self.fps = fps
        pre_frames = int(getattr(config, 'PRE_EVENT_BUFFER_SECONDS', 4) * fps)
        self.frame_buffer = deque(maxlen=max(pre_frames, 10))
        self._writer = None
        self._post_frames_remaining = 0
        self.current_clip_path = None
        self.on_clip_complete = on_clip_complete
        self._active_recording = False
        self._record_frames = []

    def push_frame(self, frame):
        """Call on every frame. Keeps the rolling pre-event buffer."""
        # Store lightweight copy in memory buffer
        self.frame_buffer.append(frame.copy())

        if self._active_recording:
            self._record_frames.append(frame.copy())
            self._post_frames_remaining -= 1

            if self._post_frames_remaining <= 0:
                self._active_recording = False
                # Offload disk encoding & dispatch to background worker thread
                frames_to_save = list(self._record_frames)
                save_path = self.current_clip_path
                fps_val = self.fps
                callback = self.on_clip_complete
                
                _ALERT_EXECUTOR.submit(self._save_video_async, frames_to_save, save_path, fps_val, callback)
                self._record_frames = []

    @staticmethod
    def _save_video_async(frames, output_path, fps, callback):
        if not frames:
            return
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"[Evidence] Clip saved asynchronously: {output_path}")

            if callback:
                callback(output_path)
        except Exception as e:
            print(f"[Evidence] Error saving clip async: {e}")

    def trigger_save(self, output_dir=config.EVIDENCE_CLIPS_DIR):
        """Triggers event recording. Merges pre-event buffer with post-event frames."""
        if self._active_recording:
            return self.current_clip_path  # Already recording this incident

        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_clip_path = os.path.join(output_dir, f"event_{timestamp_str}.mp4")

        # Snapshot current pre-event buffer
        self._record_frames = list(self.frame_buffer)
        post_secs = getattr(config, 'POST_EVENT_RECORD_SECONDS', 3)
        self._post_frames_remaining = int(post_secs * self.fps)
        self._active_recording = True

        return self.current_clip_path


class AlertDebouncer:
    """
    Intelligent Hysteresis and Cooldown Debouncer.
    
    Rules:
      1. Trigger requires sustained confidence >= ALERT_CONF_HIGH for ALERT_SUSTAINED_FRAMES.
      2. Alarm state stays active until confidence drops below ALERT_CONF_LOW (hysteresis).
      3. Enforces an ALERT_COOLDOWN_SECONDS between successive notification dispatches.
    """
    def __init__(
        self,
        conf_high=getattr(config, 'ALERT_CONF_HIGH', 0.75),
        conf_low=getattr(config, 'ALERT_CONF_LOW', 0.40),
        sustained_frames=getattr(config, 'ALERT_SUSTAINED_FRAMES', 15),
        cooldown_seconds=getattr(config, 'ALERT_COOLDOWN_SECONDS', 15.0)
    ):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.sustained_frames = sustained_frames
        self.cooldown_seconds = cooldown_seconds

        self._consecutive_high_count = 0
        self._in_alert_state = False
        self._last_alert_time = 0.0

    def update(self, confidence):
        """
        Updates debouncer state.
        
        Returns:
            bool: True ONLY on the exact frame when a new valid alert is fired.
        """
        now = time.time()

        if confidence >= self.conf_high:
            self._consecutive_high_count += 1
        elif confidence < self.conf_low:
            self._consecutive_high_count = 0
            self._in_alert_state = False

        # Check if condition to fire new alert is satisfied
        if (self._consecutive_high_count >= self.sustained_frames 
            and not self._in_alert_state 
            and (now - self._last_alert_time) >= self.cooldown_seconds):
            
            self._in_alert_state = True
            self._last_alert_time = now
            return True

        return False

    def is_alarm_active(self):
        """Returns True if current system state is considered under alarm condition."""
        return self._in_alert_state

    def reset(self):
        self._consecutive_high_count = 0
        self._in_alert_state = False


def format_alert_message(timestamp, confidence, source_id, person_count):
    """Formats HTML alert message for Telegram."""
    try:
        dt = datetime.datetime.fromisoformat(str(timestamp))
        formatted_time = dt.strftime("%A, %b %d, %Y • %I:%M:%S %p")
    except Exception:
        formatted_time = str(timestamp)

    conf_pct = f"{confidence * 100:.1f}%"

    message = (
        f"<b>🚨 SENTINEL AI — SECURITY ALERT 🚨</b>\n"
        f"<b> Incident:</b> Suspicious Behavior / Fight\n"
        f"<b> Threat Level / Conf:</b> <b>{conf_pct}</b>\n"
        f"<b> People Detected:</b> {person_count}\n"
        f"<b> Source ID:</b> <code>{source_id}</code>\n"
        f"<b> Timestamp:</b> {formatted_time}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📹 <i>Evidence video clip attached below.</i>"
    )
    return message


def _dispatch_telegram_sync(message, video_path, bot_token, chat_id, parse_mode):
    if not bot_token or not chat_id:
        return False

    # Send Video with Caption if video file exists
    if video_path and os.path.exists(video_path):
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        try:
            print(f"[Alert] Sending Telegram video alert ({os.path.basename(video_path)})...")
            with open(video_path, "rb") as video_file:
                data = {
                    "chat_id": chat_id,
                    "caption": message,
                    "parse_mode": parse_mode
                }
                files = {"video": video_file}
                response = requests.post(url, data=data, files=files, timeout=35)
                response.raise_for_status()
                print(f"[Alert] Telegram video alert dispatched successfully!")
                return True
        except Exception as e:
            print(f"[Alert] Telegram sendVideo failed: {e}")

    # Fallback / Initial Text Notification
    if message:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "text": message, "parse_mode": parse_mode},
                timeout=10
            )
            response.raise_for_status()
            print("[Alert] Telegram text alert dispatched successfully!")
            return True
        except Exception as e:
            print(f"[Alert] Telegram sendMessage failed: {e}")

    return False


def send_telegram_alert(
    message,
    video_path=None,
    bot_token=config.TELEGRAM_BOT_TOKEN,
    chat_id=config.TELEGRAM_CHAT_ID,
    parse_mode="HTML",
    async_mode=True
):
    """
    Sends formatted text alert and optional attached evidence video clip to Telegram.
    Runs asynchronously by default to avoid blocking computer vision execution.
    """
    if not bot_token or not chat_id:
        print("[Alert] Telegram bot token or chat ID not configured in .env")
        return False

    if async_mode:
        _ALERT_EXECUTOR.submit(
            _dispatch_telegram_sync,
            message, video_path, bot_token, chat_id, parse_mode
        )
        return True
    else:
        return _dispatch_telegram_sync(message, video_path, bot_token, chat_id, parse_mode)