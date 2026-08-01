from src.video_source import VideoSource
from src.detection    import PersonDetector
from src.pose         import PoseEstimator
from src.features     import FeatureEngineer
from src.classifier   import Tier2Inferencer
from src.alerting     import EventLogger, EvidenceClipWriter, AlertDebouncer, send_telegram_alert

def run_pipeline(video_source_arg):
    detector = PersonDetector()
    estimator = PoseEstimator()
    engineer = FeatureEngineer()
    inferencer = Tier2Inferencer()
    debouncer = AlertDebouncer()
    logger = EventLogger()
    clip_writer = EvidenceClipWriter()

    with VideoSource(video_source_arg) as source:
        fps = source.get_fps()
        clip_writer.fps = fps
        
        for frame in source:
            # 1. Detect people
            persons = detector.detect_and_track(frame)
            
            # 2. Estimate pose for each person
            for person in persons:
                person["keypoints"] = estimator.extract_keypoints(person["crop"])
            
            # 3. Compute features
            feature_vec = engineer.update(frame, persons)
            
            # 4. Always maintain the clip buffer (pre-event footage)
            clip_writer.push_frame(frame)
            
            if feature_vec is None:
                continue
            
            # 5. Run the classifier
            confidence = inferencer.push_features(feature_vec)
            
            if confidence is None:
                continue
            
            # 6. Debounce and fire alert
            should_alert = debouncer.update(confidence)
            
            if should_alert:
                clip_path = clip_writer.trigger_save()
                timestamp = logger.log_event(
                    confidence=confidence,
                    clip_path=clip_path,
                    source_id=str(video_source_arg),
                    person_count=len(persons)
                )
                send_telegram_alert(
                    f"⚠️ Suspicious activity detected!\n"
                    f"Time: {timestamp}\n"
                    f"Confidence: {confidence:.1%}\n"
                    f"Persons: {len(persons)}"
                )