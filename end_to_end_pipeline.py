import luigi
import os
import subprocess
import glob
import pandas as pd
import sys
import time
import re
import json
import runpy
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import ffmpeg
import imageio_ffmpeg
import cv2
import ultralytics
import pyspark

# PyInstaller creates a temporary folder at sys._MEIPASS when running.
# If running normally, it just uses the local directory.
if getattr(sys, 'frozen', False):
    APP_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_ffmpeg_cmd():
    bundled_ffmpeg = os.path.join(APP_DIR, "ffmpeg.exe")
    if os.path.exists(bundled_ffmpeg):
        return bundled_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG_CMD = resolve_ffmpeg_cmd()
FFPROBE_CMD = shutil.which("ffprobe")


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.abspath("Judo_Pipeline")
RAW_DIR = os.path.join(BASE_DIR, "01_Raw_Input")
CONVERTED_DIR = os.path.join(BASE_DIR, "02_Converted")
SEGMENTED_DIR = os.path.join(BASE_DIR, "03_Segmented")
FRAMES_DIR = os.path.join(BASE_DIR, "04_Frames")
RESULTS_DIR = os.path.join(BASE_DIR, "05_Results")
FINAL_CLIPS_DIR = os.path.join(BASE_DIR, "06_Final_Clips")

PROJECT_JSON = os.path.join(RESULTS_DIR, "project_manifest.json")
MASTER_CSV = os.path.join(RESULTS_DIR, "tournament_master_log.csv")
SEGMENT_SECONDS = int(os.getenv("JUDO_SEGMENT_SECONDS", "600"))
FAST_VIDEO_COPY = env_flag("JUDO_FAST_VIDEO_COPY", False)
FAST_FINAL_CLIP_COPY = env_flag("JUDO_FAST_FINAL_CLIP_COPY", False)
ACTION_GAP_SECONDS = float(os.getenv("JUDO_ACTION_GAP_SECONDS", "90"))

# Ensure base directories exist
for d in [RAW_DIR, CONVERTED_DIR, SEGMENTED_DIR, FRAMES_DIR, RESULTS_DIR, FINAL_CLIPS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_raw_videos():
    """Helper function to count inputs and trigger dynamic updates."""
    return list(Path(RAW_DIR).glob("*.*"))

# --- NEW: Global dictionary to store task times ---
TASK_TIMINGS = {}

# ==========================================
# 1.5 GLOBAL TASK TIMERS (Luigi Event Handlers)
# ==========================================

@luigi.Task.event_handler(luigi.Event.START)
def start_timing(task):
    """Starts the stopwatch when a task begins."""
    task._start_time = time.time()

@luigi.Task.event_handler(luigi.Event.SUCCESS)
def success_timing(task):
    """Stops the stopwatch, saves, and prints the duration when a task succeeds."""
    if hasattr(task, '_start_time'):
        elapsed_seconds = time.time() - task._start_time
        mins, secs = divmod(elapsed_seconds, 60)
        time_str = f"{int(mins)}m {secs:.1f}s"
        
        # NEW LOGIC: Store the time in our global ledger
        TASK_TIMINGS[task.__class__.__name__] = time_str
        
        print(f"\n[⏱️ TIMER] {task.__class__.__name__} completed in {time_str}")

@luigi.Task.event_handler(luigi.Event.FAILURE)
def failure_timing(task, exception):
    """Stops the stopwatch and prints the duration if a task crashes."""
    if hasattr(task, '_start_time'):
        elapsed_seconds = time.time() - task._start_time
        mins, secs = divmod(elapsed_seconds, 60)
        print(f"\n[⚠️ TIMER] {task.__class__.__name__} FAILED after {int(mins)}m {secs:.1f}s")

# ==========================================
# 2. LUIGI PIPELINE TASKS
# ==========================================

class Task1_FormatVideo(luigi.Task):
    """Formats a SINGLE video. Skips if this specific MP4 already exists."""
    video_path = luigi.Parameter()

    def output(self):
        video_stem = Path(str(self.video_path)).stem
        out_name = os.path.join(CONVERTED_DIR, f"{video_stem}_std.mp4")
        return luigi.LocalTarget(out_name)

    def run(self):
        print(f"\n>>> TASK 1: Formatting {self.video_path}...")
        if FAST_VIDEO_COPY:
            fast_cmd = [
                FFMPEG_CMD, "-y", "-i", str(self.video_path),
                "-map", "0",
                "-c", "copy",
                "-movflags", "+faststart",
                self.output().path
            ]
            result = subprocess.run(fast_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return
            print("Fast remux failed; falling back to normalized re-encode.")

        encode_cmd = [
            FFMPEG_CMD, "-y", "-i", str(self.video_path),
            "-vsync", "1",
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "22",
            "-af", "aresample=async=1",
            self.output().path
        ]
        subprocess.run(encode_cmd, check=True)


class Task2_SegmentVideos(luigi.Task):
    """Batch processes the folder, but re-runs if the raw video count changes."""
    def requires(self):
        # Demand Task 1 completes for EVERY file in the raw folder
        raw_files = get_raw_videos()
        if not raw_files:
            raise FileNotFoundError(f"Drop raw videos into {RAW_DIR} first!")
        return [Task1_FormatVideo(video_path=str(v)) for v in raw_files]

    def output(self):
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(SEGMENTED_DIR, f"_SEGMENTED_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 2: Segmenting Videos...")
        
        # Save the original terminal arguments
        original_argv = sys.argv 
        
        # Inject the arguments this specific script expects
        sys.argv = [
            "truncate_videos.py",
            "--input-root-path", CONVERTED_DIR,
            "--output-root-path", SEGMENTED_DIR,
            "--duration", str(SEGMENT_SECONDS),
        ]
        
        if os.path.abspath(".") not in sys.path:
            sys.path.insert(0, os.path.abspath("."))

        try:
            # run_module executes the script exactly as if it were called with 'python -m'
            runpy.run_path("judo_footage_analysis/workflow/truncate_videos.py", run_name="__main__")
        except SystemExit as e:
            if e.code != 0: raise # Only crash if the script actually failed
        finally:
            sys.argv = original_argv # Always restore the original arguments
            
        with self.output().open('w') as f: f.write("Done")


class Task3_ExtractFrames(luigi.Task):
    """Legacy compatibility task.

    The AI analysis reads segmented videos directly, so extracting every frame
    to JPEG is pure overhead for the current pipeline.
    """
    def requires(self): return Task2_SegmentVideos()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(FRAMES_DIR, f"_FRAMES_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 3: Skipping frame extraction (AI reads videos directly).")
        with self.output().open('w') as f: f.write("Done")


class Task4_GenerateManifest(luigi.Task):
    """Creates the JSON map required by the YOLO AI."""
    def requires(self): return Task3_ExtractFrames()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(RESULTS_DIR, f"_MANIFEST_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 4: Generating AI Manifest...")
        
        original_argv = sys.argv
        sys.argv = ["generate_combat_json.py", "--input_folder", SEGMENTED_DIR, "--output_path", PROJECT_JSON]
        
        try:
            # run_path executes a direct file exactly as if it were called with 'python file.py'
            runpy.run_path("scripts/generate_combat_json.py", run_name="__main__")
        except SystemExit as e:
            if e.code != 0: raise
        finally:
            sys.argv = original_argv
            
        with self.output().open('w') as f: f.write("Done")


class Task5_RunAIAnalysis(luigi.Task):
    """Executes the YOLOv8 model to classify combat phases."""
    def requires(self): return Task4_GenerateManifest()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(RESULTS_DIR, f"_AI_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 5: Running AI Analysis...")
        
        original_argv = sys.argv
        sys.argv = ["extract_combat_phases.py", "ExtractCombatPhases", "--project-json", PROJECT_JSON, "--output-dir", RESULTS_DIR, "--local-scheduler"]
        
        if os.path.abspath(".") not in sys.path:
            sys.path.insert(0, os.path.abspath("."))

        try:
            runpy.run_path("judo_footage_analysis/workflow/extract_combat_phases.py", run_name="__main__")
        except SystemExit as e:
            if e.code != 0: raise
        finally:
            sys.argv = original_argv
            
        with self.output().open('w') as f: f.write("Done")


class Task6_ConsolidateAndClip(luigi.Task):
    """Merges the AI data and uses FFmpeg to trim the final action clips directly from the LONG FORM video."""
    def requires(self): return Task5_RunAIAnalysis()
    
    def output(self): 
        num_vids = len(get_raw_videos())
        return luigi.LocalTarget(os.path.join(FINAL_CLIPS_DIR, f"_PIPELINE_COMPLETE_{num_vids}_FILES"))

    def run(self):
        print("\n>>> TASK 6: Consolidating Data & Clipping Matches from Master Video...")
        
        all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        all_files = [f for f in all_files if "tournament_master_log" not in f]
        
        df_list = []
        for file in all_files:
            df = pd.read_csv(file)
            chunk_name = Path(file).stem.replace(".mp4_phases", "") # e.g., "0053"
            
            # Reverse-engineer the parent video by finding where this segment lives
            segments = list(Path(SEGMENTED_DIR).rglob(f"{chunk_name}.mp4"))
            if not segments: continue
            
            parent_dir_name = segments[0].parent.name
            parent_video = parent_dir_name.replace("match_", "").replace("_std", "")
            
            # --- NEW LOGIC: Convert local chunk time into GLOBAL Master Time ---
            try:
                chunk_idx = int(chunk_name)
                offset = chunk_idx * SEGMENT_SECONDS
            except ValueError:
                offset = 0
                
            df['global_timestamp'] = df['timestamp'] + offset
            df['parent_video'] = parent_video
            df_list.append(df)
            
        if not df_list:
            raise ValueError("No CSV data found to process!")
            
        master_df = pd.concat(df_list, axis=0, ignore_index=True)
        master_df.to_csv(MASTER_CSV, index=False)

        # 2. Clip the Matches in Contiguous Blocks from the MASTER Video
        grouped = master_df.groupby('parent_video')
        for parent_video, data in grouped:
            
            # STITCHING: Sort by our new continuous global timeline!
            data = data.sort_values('global_timestamp')
            
            # Clean the text
            clean_phases = data['phase'].astype(str).str.strip().str.lower()
            inactive_phases = ['mate', 'no-match/intermission', 'none', 'nan']
            
            is_active_phase = ~clean_phases.isin(inactive_phases)
            has_fighters = data['detections'] >= 2
            is_action = is_active_phase & has_fighters
            
            # THE PATIENCE BUFFER: Bridge gaps across chunk boundaries!
            timestamp_step = data['global_timestamp'].diff().median()
            if pd.isna(timestamp_step) or timestamp_step <= 0:
                timestamp_step = 1.0
            action_gap_rows = max(1, round(ACTION_GAP_SECONDS / timestamp_step))
            is_action = is_action.replace(False, pd.NA).ffill(limit=action_gap_rows).fillna(False).astype(bool)
            
            # Reverse-Cooldown Debouncer + Crowd Density Filter
            raw_bows = data.get('bow_detected', pd.Series(False, index=data.index)).fillna(False).astype(bool)
            is_not_crowded = data['detections'] <= 8
            bows = raw_bows & is_not_crowded
            
            bow_timestamps = data.loc[bows, 'global_timestamp']
            time_to_next_bow = bow_timestamps.diff(-1).abs()
            valid_bows_mask = (time_to_next_bow > 15) | (time_to_next_bow.isna())
            
            data['valid_bow'] = False
            data.loc[bow_timestamps[valid_bows_mask].index, 'valid_bow'] = True
            data['match_id'] = data['valid_bow'].cumsum()
            
            block_changes = (is_action != is_action.shift()) | (data['match_id'] != data['match_id'].shift())
            block_ids = block_changes.cumsum()
            
            active_blocks = data[is_action].groupby(block_ids)
            
            # Set target to the 10-hour standard master video in the Converted folder
            master_video_path = os.path.join(CONVERTED_DIR, f"{parent_video}_std.mp4")
            if not os.path.exists(master_video_path):
                print(f"⚠️ Skipping {parent_video}: Master converted file not found.")
                continue

            # Figure out the exact Time of Day this master video started
            base_dt = None
            match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})", parent_video)
            if match:
                base_dt = datetime.strptime(match.group(1), "%Y-%m-%d_%H_%M_%S")
            else:
                try:
                    raw_files = list(Path(RAW_DIR).glob(f"*{parent_video}*.*"))
                    if raw_files:
                        if FFPROBE_CMD:
                            cmd = [FFPROBE_CMD, "-v", "quiet", "-show_entries", "format_tags=creation_time", "-of", "default=noprint_wrappers=1:nokey=1", str(raw_files[0])]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            creation_str = result.stdout.strip()
                            if creation_str:
                                clean_time = creation_str.split('.')[0].replace('T', ' ').replace('Z', '')
                                base_dt = datetime.strptime(clean_time, "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    pass

            # --- NEW LOGIC: Initialize match counter ---
            match_count = 1

            for block_id, block_data in active_blocks:
                if len(block_data) < 5: 
                    continue
                    
                # Calculate cut times using the GLOBAL timeline
                start_time = max(0, block_data['global_timestamp'].min() - 2)
                end_time = block_data['global_timestamp'].max() + 2
                
                if base_dt:
                    # Simply add the global seconds to the master start time
                    clip_dt = base_dt + timedelta(seconds=start_time)
                    time_str = clip_dt.strftime("%Hh%Mm%Ss")
                else:
                    hrs, rem = divmod(int(start_time), 3600)
                    mins, secs = divmod(rem, 60)
                    time_str = f"elapsed_{hrs:02d}h{mins:02d}m{secs:02d}s"
                
                # --- NEW LOGIC: Insert Match Number ---
                clip_filename = f"{parent_video}_Match_{match_count:02d}_AT_{time_str}.mp4"
                
                out_path = os.path.join(FINAL_CLIPS_DIR, clip_filename)
                print(f"✂ Clipping Match {match_count:02d} -> {start_time:.1f}s to {end_time:.1f}s (Saved as {clip_filename})")
                
                duration = max(0.1, end_time - start_time)
                if FAST_FINAL_CLIP_COPY:
                    fast_cmd = [
                        FFMPEG_CMD, "-y",
                        "-ss", str(start_time),
                        "-i", master_video_path,
                        "-t", str(duration),
                        "-c", "copy",
                        "-avoid_negative_ts", "make_zero",
                        out_path
                    ]
                    result = subprocess.run(fast_cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        match_count += 1
                        continue
                    print("Fast clip copy failed; falling back to accurate re-encode.")

                # Put -ss after -i for accurate cuts. This is slower, but keeps clip
                # boundaries closer to the AI timestamps.
                encode_cmd = [
                    FFMPEG_CMD, "-y",
                    "-i", master_video_path,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-c:a", "aac",
                    out_path
                ]
                subprocess.run(encode_cmd, check=True)
                
                # --- NEW LOGIC: Increment the counter for the next loop ---
                match_count += 1
                
        with self.output().open('w') as f: f.write("Done")

# ==========================================
# 3. TRIGGER
# ==========================================
if __name__ == "__main__":
    print("\n>>> STARTING JUDO PIPELINE <<<")
    start_time = time.time()
    
    # Pointing Luigi to the final task causes it to chain everything else automatically
    luigi.build([Task6_ConsolidateAndClip()], workers=1, local_scheduler=False)
    
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    
    # --- NEW LOGIC: Print the Final Summary Report ---
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    
    if not TASK_TIMINGS:
        print(" No new tasks were executed (all outputs already exist).")
    else:
        for task_name, duration in TASK_TIMINGS.items():
            # This formatting pads the task name with spaces so all the times align perfectly
            print(f" {task_name.ljust(25)} : {duration}")
            
    print("-" * 50)
    print(f" TOTAL EXECUTION TIME      : {total_minutes:.2f} minutes")
    print("="*50 + "\n")
