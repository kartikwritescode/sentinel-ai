import cv2
import time
import threading
from queue import Queue, Empty


class VideoSource:
    """
    A high-performance unified wrapper around any video input.
    
    Supports:
      - Webcam (0, 1, ...)
      - IP Camera / Phone Stream (http://...)
      - RTSP Surveillance Camera (rtsp://...)
      - Local Video Files (clip.mp4, clip.avi)

    Features:
      - Multi-threaded background capture worker for live streams (prevents frame queue lag)
      - Automatic frame dropping if downstream inference is slower than camera FPS
      - Auto-cleanup on exit
    """
    def __init__(self, source, threaded=None):
        self.source = source
        self._is_live = isinstance(source, int) or (isinstance(source, str) and (source.isdigit() or source.startswith(("http://", "https://", "rtsp://"))))
        self.threaded = self._is_live if threaded is None else threaded
        
        self._cap = None
        self._running = False
        self._thread = None
        self._frame_queue = Queue(maxsize=2)
        self._latest_frame = None
        self._fps = 30.0

    def open(self):
        # On Windows, DirectShow backend (CAP_DSHOW) opens webcams 5x faster with lower latency
        src_val = int(self.source) if (isinstance(self.source, str) and self.source.isdigit()) else self.source
        
        if isinstance(src_val, int):
            self._cap = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                self._cap = cv2.VideoCapture(src_val)
        else:
            self._cap = cv2.VideoCapture(src_val)

        if not self._cap.isOpened():
            raise RuntimeError(f"Couldn't open video source: {self.source}")

        fps_val = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps_val if (fps_val and fps_val > 0 and fps_val < 120) else 30.0

        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._thread.start()

        return self

    def _capture_worker(self):
        """Dedicated background thread continuously draining camera hardware buffer."""
        while self._running and self._cap.isOpened():
            success, frame = self._cap.read()
            if not success:
                break
            
            # Keep newest frame; drop older frame if consumer is busy
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass

        self._running = False

    def read_frame(self):
        """Returns (success: bool, frame: np.ndarray)"""
        if self.threaded:
            if not self._running and self._frame_queue.empty():
                return False, None
            try:
                # Wait up to 1 second for next frame from worker
                frame = self._frame_queue.get(timeout=1.0)
                return True, frame
            except Empty:
                return False, None
        else:
            if self._cap is None:
                return False, None
            return self._cap.read()

    def get_fps(self):
        return self._fps

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        while True:
            success, frame = self.read_frame()
            if not success or frame is None:
                break
            yield frame