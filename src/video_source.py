# unified open any video abstraction

import cv2

class VideoSource:
    """
    A unified wrapper around any video input.
    
    Usage:
        source = VideoSource(0)                          # webcam
        source = VideoSource("http://192.168.1.5:8080/video")  # phone IP cam
        source = VideoSource("rtsp://...")               # RTSP stream
        source = VideoSource("path/to/clip.mp4")         # uploaded file
        
        with source:
            for frame in source:
                # frame is a numpy array (H, W, 3) in BGR format
                process(frame)
    """
    def __init__(self,source):
        self.source = source
        self.cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Couldn't open video source: {self.source}")
        return self

    def read_frame(self):
        '''Returns (success: bool, frame: np.ndarray)'''
        return self._cap.read()

    def get_fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def release(self):
        if self._cap:
            self._cap.release()        

    # will let us use 'with VideoSource(...) as src:

    def __enter__(self):
        return self.open()

    def __exit__(self,*args):
        self.release()

    # lets you do 'for frame in source:'
    def __iter__(self):
        while True:
            success , frame = self.read_frame()
            if not success:
                break
            yield frame



# open() returns an obj -> that implements context manager , which further has 2 special commands __enter__() , __exit__()
# enter() -> Runs before the code inside with.
# exit() -> Runs after leaving the block
# to prevent resource leak (keeping the file open)




# __iter__ with yield -> instead of loading all frames at the same time , the yields gives u one frame at a time