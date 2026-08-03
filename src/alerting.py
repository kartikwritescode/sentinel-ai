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
    
    SQLite is a file-based database — no server, no setup, just a .db file.
    It's built into Python's standard library.
    """
    def __init__(self,db_path = config.SQLITE_DB_PATH):
        os.makedirs(os.path.dirname(db_path),exist_ok=True)
        self.conn=sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        '''Create the events table if it doesn't exist yet.'''
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

    def log_event(self,confidence,clip_path=None,source_id = 'unknown',person_count=0):
        timestamp = datetime.datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO events(timestamp , confidence , clip_path, source_id, person_count) "
            "VALUES(?,?,?,?,?)",
            (timestamp, confidence, clip_path, source_id, person_count)  # was 'time', fixed to 'timestamp'
        )
        self.conn.commit()
        return timestamp

    def get_recent_events(self,limit=50):
        cursor = self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)  # was FROOM, fixed
        )
        return cursor.fetchall()
    def close(self):
        self.conn.close()

class EvidenceClipWriter:
    """
    Saves a video clip that includes footage BEFORE the alert triggered.
    
    This is the 'pre-event buffer' pattern. We keep a rolling deque of the
    last N seconds. When an alert fires, we dump the buffer + a few more
    seconds into a file.
    """
    
    def __init__(self, fps=30):
        self.fps = fps
        pre_frames = int(config.PRE_EVENT_BUFFER_SECONDS * fps)
        self.frame_buffer = deque(maxlen=pre_frames)
        self._writer = None
        self._post_frames_remaining = 0
        self.current_clip_path = None
    def push_frame(self , frame):
        """Call this every frame, always. It maintains the rolling buffer."""
        self.frame_buffer.append(frame.copy())


        # for actively recording post event footage , write to file
        if self._writer is not None:
            self._writer.write(frame)
            self._post_frames_remaining -=1

            if self._post_frames_remaining <= 0:
                self._writer.release()
                self._writer = None
                print(f"Evidence clip saved:{self.current_clip_path}")

    def trigger_save(self, output_dir=config.EVIDENCE_CLIPS_DIR):
        """Call this when the alert fires. Dumps buffer + post-event to disk."""
        if self._writer is not None:
            return  # already recording, don't start again
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_clip_path = os.path.join(output_dir, f"event_{timestamp_str}.mp4")
        
        if self.frame_buffer:
            h, w = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(
                self.current_clip_path, fourcc, self.fps, (w, h)
            )
            # Write the buffered pre-event frames first
            for f in self.frame_buffer:
                self._writer.write(f)
        
        self._post_frames_remaining = int(config.POST_EVENT_RECORD_SECONDS * self.fps)
        return self.current_clip_path


class AlertDebouncer:
    """
    Prevents false alarms from single-frame glitches.
    
    The model must return 'suspicious' for N CONSECUTIVE windows before
    we actually fire an alert. One bad prediction in a sea of normal ones
    gets ignored.
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
        """
        Args:
            confidence: float -> model's output (0.0 to 1.0)
        
        Returns:
            True if an alert should fire, False otherwise
        """
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


def send_telegram_alert(message, bot_token=config.TELEGRAM_BOT_TOKEN, chat_id=config.TELEGRAM_CHAT_ID):
    """
    Sends a text message to your Telegram chat.
    Replace with Discord webhook if you prefer — same HTTP POST pattern.
    """
    if not bot_token or not chat_id:
        print("[Alert] Telegram not configured — skipping notification")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[Alert] Telegram send failed: {e}")