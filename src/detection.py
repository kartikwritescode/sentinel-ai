# yolo person detection

from ultralytics import YOLO
import config

class PersonDetector:
    """
    Wraps YOLO11n for real-time person detection + ByteTrack tracking.
    
    ByteTrack is built into Ultralytics -> calling model.track() instead of
    model.predict() activates it automatically. It assigns each person a
    persistent integer ID that stays the same across frames, even if they
    briefly leave the frame.
    """

    def __init__(self):
        # load the model once at startup -> expensive operation so better to not keep it inside loop
        self.model = YOLO(config.YOLO_MODEL)
        self.conf = YOLO(config.YOLO_CONF_THRESH)
        self.classes = YOLO(config.YOLO_CLASSES)

    def detect_and_track(self,frame):




        results = self.model.track(
            source=frame,
            traker = 'bytetrack.yaml', 
            classes = self.classes,  #only persons(class 0)
            conf = self.conf,   
            persist = True,          #keeps track ids stable -> without this ByteTrack resets its memory every frame
            verbose=False
        )

        persons = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                track_id = int(box.id) if box.id is not None else -1
                x1 , y1 , x2 , y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                # crop the person outta frame for pose estimation
                crop = frame[y1:y2 , x1:x2]

                persons.append({
                    'track_id':track_id,
                    'bbox':[x1,y1,x2,y2],
                    'confidence':confidence,
                    'crop':crop
                })
        return persons