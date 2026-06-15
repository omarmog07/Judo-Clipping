import luigi
import json
import os
import cv2
import pandas as pd
import ssl
from ultralytics import YOLO

# Fix for SSL Certificate Verification errors on Windows
ssl._create_default_https_context = ssl._create_unverified_context


class ExtractCombatPhases(luigi.Task):
    project_json = luigi.Parameter()
    output_dir = luigi.Parameter()
    sample_interval_seconds = luigi.FloatParameter(default=float(os.getenv("JUDO_SAMPLE_INTERVAL_SECONDS", "1.0")))
    batch_size = luigi.IntParameter(default=int(os.getenv("JUDO_YOLO_BATCH_SIZE", "16")))
    confidence = luigi.FloatParameter(default=float(os.getenv("JUDO_YOLO_CONF", "0.15")))

    def output(self):
        # Create a success flag file for Luigi
        return luigi.LocalTarget(os.path.join(self.output_dir, "_SUCCESS"))

    def run(self):
        # 1. Load the "Map" created by the generator script
        with open(self.project_json, 'r') as f:
            videos = json.load(f)

        # 2. Load the YOLOv8 model
        # It will auto-download 'yolov8n.pt' on the first run
        model = YOLO('judo_custom_v3.pt')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for video in videos:
            # INTEGRATED FIX: Match the "video" key from your specific JSON
            v_path = video.get('video') or video.get('video_path') or video.get('path')

            # Ensure we have a valid path before continuing
            if not v_path:
                print(f"Skipping entry: No video path found in {video}")
                continue

            # Get name from JSON or extract from filename
            v_name = video.get('video_name') or video.get('name') or os.path.basename(v_path)

            print(f"Processing: {v_name}")

            cap = cv2.VideoCapture(v_path)
            results_data = []
            frame_count = 0
            pending_frames = []
            pending_timestamps = []

            if not cap.isOpened():
                print(f"Error: Could not open video {v_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30

            sample_stride = max(1, round(fps * float(self.sample_interval_seconds)))

            def flush_batch():
                nonlocal pending_frames, pending_timestamps
                if not pending_frames:
                    return

                batch_results = model(
                    pending_frames,
                    conf=float(self.confidence),
                    verbose=False,
                    batch=int(self.batch_size),
                )

                for timestamp, results in zip(pending_timestamps, batch_results):
                    classes_in_frame = results.boxes.cls.cpu().tolist()
                    bow_detected = (1.0 in classes_in_frame) or (1 in classes_in_frame)
                    phase = self.classify_phase(results.boxes)

                    results_data.append({
                        "timestamp": timestamp,
                        "phase": phase,
                        "detections": len(results.boxes),
                        "bow_detected": bow_detected
                    })

                pending_frames = []
                pending_timestamps = []

            while cap.isOpened():
                ret = cap.grab()
                if not ret: break

                if frame_count % sample_stride == 0:
                    retrieved, frame = cap.retrieve()
                    if not retrieved:
                        break
                    pending_frames.append(frame)
                    pending_timestamps.append(frame_count / fps)
                    if len(pending_frames) >= int(self.batch_size):
                        flush_batch()

                frame_count += 1

            flush_batch()

            # Save the results for this video
            df = pd.DataFrame(results_data)
            output_file = os.path.join(self.output_dir, f"{v_name}_phases.csv")
            df.to_csv(output_file, index=False)
            cap.release()

        # Mark the entire Luigi task as finished
        with self.output().open('w') as f:
            f.write("Completed Successfully")

    def classify_phase(self, boxes):
        """Heuristic logic to distinguish standing from groundwork"""
        if len(boxes) < 2:
            return "No-Match/Intermission"

        try:
            # xywh[0][3] is the height of the bounding box
            heights = [b.xywh[0][3].item() for b in boxes]
            avg_height = sum(heights) / len(heights)

            # Threshold of 150 pixels (adjustable based on camera distance)
            return "Tachi-waza" if avg_height > 150 else "Ne-waza"
        except Exception:
            return "Unknown"


if __name__ == "__main__":
    luigi.run()
