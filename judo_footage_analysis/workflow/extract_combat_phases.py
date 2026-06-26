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

            if not cap.isOpened():
                print(f"Error: Could not open video {v_path}")
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Sample every 30 frames (approx 1 frame per second)
                if frame_count % 30 == 0:
                   # ML Inference
                    results = model(frame, conf=0.15, verbose=False)[0]

                    # NEW LOGIC: Check if Match_Start (Class ID 1) is detected
                    classes_in_frame = results.boxes.cls.cpu().tolist()
                    bow_detected = (1.0 in classes_in_frame) or (1 in classes_in_frame)

                    # Classify if the frame is Standing (Tachi-waza) or Groundwork (Ne-waza)
                    phase = self.classify_phase(results.boxes)

                    results_data.append({
                        "timestamp": frame_count / 30,
                        "phase": phase,
                        "detections": len(results.boxes),
                        "bow_detected": bow_detected  # Added to the CSV log
                    })

                frame_count += 1

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