import cv2
import numpy as np
from src.video_source import VideoSource
from src.detection    import PersonDetector
from src.features     import FeatureEngineer
from src.classifier   import Tier2Inferencer
from src.alerting     import EventLogger, EvidenceClipWriter, AlertDebouncer, send_telegram_alert

# COCO 17 Keypoint Skeleton connection pairs
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face / Ears
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left Arm
    (6, 8), (8, 10),                     # Right Arm
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15),                  # Left Leg
    (12, 14), (14, 16)                   # Right Leg
]

def draw_visual_overlays(frame, persons, confidence, is_alerting):
    """
    Draws professional CCTV overlays:
    - Bounding boxes & Track IDs
    - Pose Skeleton joints and bones
    - Top HUD Status Banner (Normal / Warning / Alarm)
    """
    h, w = frame.shape[:2]

    # 1. Draw Pose Skeletons & Bounding Boxes
    for person in persons:
        track_id = person.get('track_id', -1)
        bbox     = person.get('bbox', [0, 0, 0, 0])
        kp       = person.get('keypoints')

        # Box Color: Red if alerting, Green otherwise
        box_color = (0, 0, 255) if is_alerting else (0, 255, 0)
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"ID #{track_id}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Draw Keypoints and Skeleton bones
        if kp is not None and len(kp) >= 51:
            pts = []
            for idx in range(17):
                px = int(kp[idx * 3])
                py = int(kp[idx * 3 + 1])
                conf = kp[idx * 3 + 2]
                pts.append((px, py, conf))

            # Draw bones
            for p1_idx, p2_idx in SKELETON_CONNECTIONS:
                x_a, y_a, c_a = pts[p1_idx]
                x_b, y_b, c_b = pts[p2_idx]
                if c_a > 0.3 and c_b > 0.3 and x_a > 0 and x_b > 0:
                    cv2.line(frame, (x_a, y_a), (x_b, y_b), (255, 255, 0), 2)

            # Draw joint dots
            for px, py, conf in pts:
                if conf > 0.3 and px > 0 and py > 0:
                    cv2.circle(frame, (px, py), 4, (0, 165, 255), -1)

    # 2. Draw Top HUD Banner
    banner_color = (0, 0, 200) if is_alerting else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (w, 45), banner_color, -1)

    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "Analyzing..."

    if is_alerting:
        status_text = f"ALERT: SUSPICIOUS BEHAVIOR DETECTED! ({conf_str})"
        text_color = (0, 255, 255)
    elif confidence is not None and confidence >= 0.5:
        status_text = f"WARNING: ELEVATED MOTION ({conf_str})"
        text_color = (0, 165, 255)
    else:
        status_text = f"SYSTEM NORMAL | Conf: {conf_str} | People: {len(persons)}"
        text_color = (0, 255, 0)

    cv2.putText(frame, status_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def run_pipeline(video_source_arg, display=True):
    """
    Main CCTV Pipeline Entry Point.

    Args:
        video_source_arg: 0 (for webcam) or path to video file (e.g. 'clip.mp4')
        display: bool — if True, opens a real-time OpenCV desktop window
    """
    print(f"\n[CCTV Pipeline] Starting video source: {video_source_arg}")
    print("[CCTV Pipeline] Press 'q' in the video window to stop.\n")

    detector   = PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = Tier2Inferencer()
    debouncer  = AlertDebouncer()
    logger     = EventLogger()
    clip_writer = EvidenceClipWriter()

    window_name = "Smart CCTV - Real-Time Surveillance Feed (Press 'q' to exit)"

    latest_confidence = None
    is_alerting = False

    with VideoSource(video_source_arg) as source:
        fps = source.get_fps()
        clip_writer.fps = fps

        for frame in source:
            # 1. Detect people + pose on GPU in a single pass
            persons = detector.detect_and_track(frame)

            # 2. Compute motion & interaction features
            feature_vec = engineer.update(frame, persons)

            # 3. Always maintain pre-event video clip buffer
            clip_writer.push_frame(frame)

            if feature_vec is not None:
                # 4. Infer suspicion probability using Tier 2 GRU
                conf = inferencer.push_features(feature_vec)
                if conf is not None:
                    latest_confidence = conf
                    # 5. Debounce alerts (requires N consecutive positive windows)
                    is_alerting = debouncer.update(latest_confidence)

                    if is_alerting:
                        clip_path = clip_writer.trigger_save()
                        timestamp = logger.log_event(
                            confidence=latest_confidence,
                            clip_path=clip_path,
                            source_id=str(video_source_arg),
                            person_count=len(persons)
                        )
                        send_telegram_alert(
                            f"⚠️ Suspicious activity detected!\n"
                            f"Time: {timestamp}\n"
                            f"Confidence: {latest_confidence:.1%}\n"
                            f"Persons: {len(persons)}"
                        )

            # 6. Render visual overlays and display desktop window
            if display:
                draw_visual_overlays(frame, persons, latest_confidence, is_alerting)
                cv2.imshow(window_name, frame)

                # Exit if user presses 'q' or closes window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[CCTV Pipeline] Stopping feed.")
                    break

        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    # convert digit strings to int for webcam index (e.g. "0" -> 0)
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    run_pipeline(src)