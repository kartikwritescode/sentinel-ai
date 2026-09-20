import time
import cv2
import numpy as np
from src.video_source import VideoSource
from src.detection    import PersonDetector
from src.features     import FeatureEngineer
from src.classifier   import Tier2Inferencer
from src.alerting     import EventLogger, EvidenceClipWriter, AlertDebouncer, send_telegram_alert, format_alert_message
import config

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

def draw_visual_overlays(frame, persons, confidence, is_alerting, fps_val=0.0):
    """
    Draws professional CCTV overlays:
    - Bounding boxes & Track IDs
    - Pose Skeleton joints and bones
    - Top HUD Status Banner (Normal / Warning / Alarm) with FPS & person count
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
                if c_a > 0.25 and c_b > 0.25 and x_a > 0 and x_b > 0:
                    cv2.line(frame, (x_a, y_a), (x_b, y_b), (255, 255, 0), 2)

            # Draw joint dots
            for px, py, conf in pts:
                if conf > 0.25 and px > 0 and py > 0:
                    cv2.circle(frame, (px, py), 4, (0, 165, 255), -1)

    # 2. Draw Top HUD Banner
    banner_color = (0, 0, 200) if is_alerting else (35, 35, 35)
    cv2.rectangle(frame, (0, 0), (w, 45), banner_color, -1)

    conf_str = f"{confidence * 100:.1f}%" if confidence is not None else "0.0%"

    if is_alerting:
        status_text = f"ALERT: VIOLENT FIGHT DETECTED! ({conf_str}) | {fps_val:.1f} FPS"
        text_color = (0, 255, 255)
    elif confidence is not None and confidence >= 0.60:
        status_text = f"WARNING: ELEVATED INTERACTION ({conf_str}) | {fps_val:.1f} FPS"
        text_color = (0, 165, 255)
    elif len(persons) == 0:
        status_text = f"SYSTEM NORMAL (NO PERSONS DETECTED) | {fps_val:.1f} FPS"
        text_color = (0, 255, 0)
    elif len(persons) == 1:
        status_text = f"SYSTEM NORMAL (1 PERSON DETECTED) | Conf: {conf_str} | {fps_val:.1f} FPS"
        text_color = (0, 255, 0)
    else:
        status_text = f"SYSTEM NORMAL | Conf: {conf_str} | People: {len(persons)} | {fps_val:.1f} FPS"
        text_color = (0, 255, 0)

    cv2.putText(frame, status_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)


def run_pipeline(video_source_arg, display=True):
    """
    Main CCTV Pipeline Entry Point tailored for high-speed 60+ FPS execution.
    """
    print(f"\n[CCTV Pipeline] Starting video source: {video_source_arg}")
    print("[CCTV Pipeline] Press 'q' in the video window to stop.\n")

    detector   = PersonDetector()
    engineer   = FeatureEngineer()
    inferencer = Tier2Inferencer()
    debouncer  = AlertDebouncer(
        conf_high=getattr(inferencer, 'calibrated_threshold', getattr(config, 'ALERT_CONF_HIGH', 0.65)),
        conf_low=getattr(config, 'ALERT_CONF_LOW', 0.35),
        sustained_frames=getattr(config, 'ALERT_SUSTAINED_FRAMES', 5)
    )
    logger     = EventLogger()
    clip_writer = EvidenceClipWriter()

    window_name = "Smart CCTV - Sentinel AI Real-Time Surveillance (Press 'q' to exit)"

    latest_confidence = 0.0
    fps_display = 60.0
    frame_times = []
    frame_idx = 0
    persons = []

    with VideoSource(video_source_arg, threaded=True) as source:
        fps = source.get_fps()
        clip_writer.fps = fps

        for frame in source:
            frame_idx += 1
            t0 = time.time()

            # 1. GPU Pose Detection & Tracking (run every 2nd frame, hold on alternate frames for 60-80 FPS)
            if frame_idx % 2 == 1 or len(persons) == 0:
                persons = detector.detect_and_track(frame)

            # 2. Compute motion & interaction features
            feature_vec = engineer.update(frame, persons)

            # 3. Maintain rolling pre-event video clip buffer
            clip_writer.push_frame(frame)

            # 4. Infer suspicion probability using Unified Inferencer (with production person gating)
            if len(persons) == 0:
                conf = 0.0
            elif getattr(inferencer, 'model_type', '') == 'vision_bilstm':
                conf = inferencer.push_frame(frame, persons=persons)
            elif getattr(inferencer, 'model_type', '') == 'hybrid':
                conf_v = inferencer.push_frame(frame, persons=persons)
                conf_p = inferencer.push_features(feature_vec, person_count=len(persons)) if feature_vec is not None else None
                conf = max(conf_v or 0.0, conf_p or 0.0) if (conf_v or conf_p) else None
            else:
                conf = inferencer.push_features(feature_vec, person_count=len(persons)) if feature_vec is not None else None

            if conf is not None:
                # Exponential moving average smoothing for stable HUD readout
                latest_confidence = 0.60 * latest_confidence + 0.40 * conf

                # 5. Debounce with hysteresis & cooldown
                new_alert_triggered = debouncer.update(latest_confidence, person_count=len(persons))

                if new_alert_triggered:
                    clip_path = clip_writer.trigger_save()
                    timestamp = logger.log_event(
                        confidence=latest_confidence,
                        clip_path=clip_path,
                        source_id=str(video_source_arg),
                        person_count=len(persons)
                    )
                    formatted_msg = format_alert_message(
                        timestamp=timestamp,
                        confidence=latest_confidence,
                        source_id=str(video_source_arg),
                        person_count=len(persons)
                    )

                    # Set async callback on clip completion for Telegram video dispatch
                    def make_send_callback(msg):
                        def _callback(saved_video_path):
                            send_telegram_alert(message=msg, video_path=saved_video_path, async_mode=True)
                        return _callback

                    clip_writer.on_clip_complete = make_send_callback(formatted_msg)

                    # Send immediate non-blocking text notification
                    send_telegram_alert(message=formatted_msg, async_mode=True)

            is_alarm_active = debouncer.is_alarm_active()

            # Calculate rolling real-time FPS
            t1 = time.time()
            frame_times.append(t1 - t0)
            if len(frame_times) > 20:
                frame_times.pop(0)
            avg_dt = np.mean(frame_times) if frame_times else 0.016
            fps_display = 1.0 / avg_dt if avg_dt > 0 else 60.0

            # 6. Render visual overlays and display desktop window
            if display:
                draw_visual_overlays(frame, persons, latest_confidence, is_alarm_active, fps_display)
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