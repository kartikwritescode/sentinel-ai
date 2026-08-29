# telegram + sqlite logging

import sqlite3
import os
import datetime
import requests
import cv2
from collections import deque
import config


class EventLogger:
    """
    Logs detected incidents to a SQLite database.
    """
    def __init__(self, db_path=config.SQLITE_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Create the events table if it doesn't exist yet."""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                clip_path   TEXT,
                source_id   TEXT,
                person_count INTEGER
            )
        """)
        self.conn.commit()

    def log_event(self, confidence, clip_path=None, source_id='unknown', person_count=0):
        timestamp = datetime.datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO events(timestamp, confidence, clip_path, source_id, person_count) "
            "VALUES(?,?,?,?,?)",
            (timestamp, confidence, clip_path, source_id, person_count)
        )
        self.conn.commit()
        return timestamp

    def get_recent_events(self, limit=50):
        cursor = self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()


class EvidenceClipWriter:
    """
    Saves a video clip that includes footage BEFORE and AFTER the alert triggered.
    Triggers callback on_clip_complete(clip_path) when video file is closed.
    """
    def __init__(self, fps=30, on_clip_complete=None):
        self.fps = fps
        pre_frames = int(config.PRE_EVENT_BUFFER_SECONDS * fps)
        self.frame_buffer = deque(maxlen=pre_frames)
        self._writer = None
        self._post_frames_remaining = 0
        self.current_clip_path = None
        self.on_clip_complete = on_clip_complete

    def push_frame(self, frame):
        """Call this every frame, always. It maintains the rolling buffer."""
        self.frame_buffer.append(frame.copy())

        # for actively recording post event footage, write to file
        if self._writer is not None:
            self._writer.write(frame)
            self._post_frames_remaining -= 1

            if self._post_frames_remaining <= 0:
                self._writer.release()
                self._writer = None
                saved_path = self.current_clip_path
                print(f"[Evidence] Clip saved to disk: {saved_path}")

                # Fire completion callback (e.g. send video to Telegram)
                if self.on_clip_complete:
                    try:
                        self.on_clip_complete(saved_path)
                    except Exception as e:
                        print(f"[Evidence] Completion callback error: {e}")

    def trigger_save(self, output_dir=config.EVIDENCE_CLIPS_DIR):
        """Call this when the alert fires. Dumps buffer + post-event to disk."""
        if self._writer is not None:
            return self.current_clip_path  # already recording

        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_clip_path = os.path.join(output_dir, f"event_{timestamp_str}.mp4")

        if self.frame_buffer:
            h, w = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(
                self.current_clip_path, fourcc, self.fps, (w, h)
            )
            for f in self.frame_buffer:
                self._writer.write(f)

        self._post_frames_remaining = int(config.POST_EVENT_RECORD_SECONDS * self.fps)
        return self.current_clip_path


class AlertDebouncer:
    """
    Prevents false alarms from single-frame glitches.
    Requires N consecutive positive windows before firing alert.
    """
    def __init__(
        self,
        n_consecutive=config.ALERT_DEBOUNCE_WINDOWS,
        confidence_thresh=config.ALERT_CONFIDENCE_THRESH
    ):
        self.n_consecutive     = n_consecutive
        self.confidence_thresh = confidence_thresh
        self._consecutive_count = 0
        self._alert_active      = False

    def update(self, confidence):
        if confidence >= self.confidence_thresh:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
            self._alert_active = False

        if self._consecutive_count >= self.n_consecutive and not self._alert_active:
            self._alert_active = True
            return True

        return False

    def reset(self):
        self._consecutive_count = 0
        self._alert_active = False


def format_alert_message(timestamp, confidence, source_id, person_count):
    """
    Format a clean, highly readable HTML alert message for Telegram.
    """
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


def send_telegram_alert(
    message,
    video_path=None,
    bot_token=config.TELEGRAM_BOT_TOKEN,
    chat_id=config.TELEGRAM_CHAT_ID,
    parse_mode="HTML"
):
    """
    Sends formatted text alert and optional attached evidence video clip to Telegram.
    """
    if not bot_token or not chat_id:
        print("[Alert] Telegram bot token or chat ID not set in .env — skipping notification")
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
                response = requests.post(url, data=data, files=files, timeout=30)
                response.raise_for_status()
                print(f"[Alert] Telegram video alert dispatched successfully!")
                return True
        except requests.RequestException as e:
            print(f"[Alert] Telegram sendVideo failed: {e}")

    # Fallback: Send Text Message if no video or video send failed
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
        except requests.RequestException as e:
            print(f"[Alert] Telegram sendMessage failed: {e}")

    return False