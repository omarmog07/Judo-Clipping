"""
Workflow for sampling frames from livestream judo videos.

This script truncates videos into smaller segments for analysis.
It uses imageio-ffmpeg to locate the binary and bypasses the need for ffprobe.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import math
import subprocess
import re
import ffmpeg
import luigi
import imageio_ffmpeg
from judo_footage_analysis.utils import ensure_path

# --- CONFIGURATION: Locate FFmpeg Binary ---
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using FFmpeg: {FFMPEG_PATH}")
# -------------------------------------------

def get_duration_ffmpeg(video_path, ffmpeg_path):
    """
    Retrieves video duration using ffmpeg instead of ffprobe.
    """
    cmd = [ffmpeg_path, "-i", str(video_path)]
    # ffmpeg prints file info to stderr
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Regex to extract 'Duration: 00:00:00.00'
    match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", result.stderr)
    if match:
        hours, minutes, seconds = match.groups()
        total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        return total_seconds
    
    raise ValueError(f"Could not extract duration from video: {video_path}")

class TruncateVideos(luigi.Task):
    input_path = luigi.Parameter()
    output_root = luigi.Parameter()
    output_prefix = luigi.Parameter(default="match")

    offset = luigi.IntParameter(default=0)
    duration = luigi.IntParameter(default=600)
    clips_per_folder = luigi.IntParameter(default=6)

    @property
    def output_path(self):
        base_name = Path(self.input_path).stem
        return Path(self.output_root) / f"{self.output_prefix}_{base_name}"

    def output(self):
        return luigi.LocalTarget(self.output_path / "_SUCCESS")

    def run(self):
        out_dir = ensure_path(self.output_path)

        # FIX: Use custom function to get duration via ffmpeg.exe
        try:
            total_duration = get_duration_ffmpeg(self.input_path, FFMPEG_PATH)
        except Exception as e:
            print(f"Error getting duration: {e}")
            raise

        truncations = max(1, math.ceil(total_duration / self.duration))

        for i in range(truncations):
            start_time = self.offset + i * self.duration
            if start_time >= total_duration:
                break

            output_file = out_dir / f"{i:04d}.mp4"
            try:
                (
                    ffmpeg.input(self.input_path, ss=start_time, t=self.duration)
                    .output(str(output_file), vcodec='copy', acodec='copy', format='mp4')
                    .run(overwrite_output=True, capture_stdout=True, capture_stderr=True, cmd=FFMPEG_PATH)
                )
            except ffmpeg.Error as e:
                print(f"Fast segment copy failed for segment {i}; falling back to re-encode.")
                try:
                    (
                        ffmpeg.input(self.input_path, ss=start_time, t=self.duration)
                        .output(str(output_file), vcodec='libx264', acodec='aac', preset='veryfast', crf=24, format='mp4')
                        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True, cmd=FFMPEG_PATH)
                    )
                except ffmpeg.Error as fallback_error:
                    print(f"FFmpeg failed for segment {i}:")
                    if fallback_error.stderr:
                        print(fallback_error.stderr.decode())
                    raise

        with self.output().open("w") as f:
            f.write("")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root-path", type=str, required=True)
    parser.add_argument("--output-root-path", type=str, required=True)
    parser.add_argument("--output-prefix", type=str, default="match")
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--clips-per-folder", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Case-insensitive check for .mp4 files
    videos = [p for p in Path(args.input_root_path).glob("*") if p.suffix.lower() == ".mp4"]
    videos = sorted(videos)

    tasks = []
    for v in videos:
        tasks.append(
            TruncateVideos(
                input_path=str(v),
                output_root=str(args.output_root_path),
                output_prefix=args.output_prefix,
                duration=args.duration,
                clips_per_folder=args.clips_per_folder,
            )
        )

    luigi.build(tasks, workers=1, local_scheduler=True)
