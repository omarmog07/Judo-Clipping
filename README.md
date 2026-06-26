# AI Highlight Clipper (v1.1.9)

An automated, event-driven video processing pipeline. This software ingests long-form tournament footage and utilizes a custom YOLOv8 model to automatically cut match clips using Luigi and FFmpeg. 

**If you are looking for a ready-to-run app; download, unzip, and run the .exe: [Google Drive Link](https://drive.google.com/file/d/1wNeVxSq_D55Yg9EKilDD0E8N_wfdeunV/view?usp=sharing)**

<img width="664" height="388" alt="gui_image" src="https://github.com/user-attachments/assets/804dd017-5989-4437-ac15-cdd32e104e71" />

## Setup Instructions

### 1. Clone the Repository
Ensure you use the `.git` clone URL provided by GitHub's code button, rather than copying the browser window title URL.
```bash
git clone https://github.com/ethanrpowell/judo_tournament_video_clipper
cd judo_tournament_video_clipper
```

### 2. Configure the Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install FFmpeg
The pipeline relies on FFmpeg for frame extraction and lossless video clipping.
1. Download the Windows essential build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the archive and copy `ffmpeg.exe` and `ffprobe.exe` directly into the root folder of this repository (next to `app.py`).

### 4. Execution
With the environment activated and binaries in place, launch the UI natively:
```bash
python app.py
```
To compile a standalone executable, run:
```bash
pyinstaller --noconfirm --onedir --console --collect-all pyspark --collect-all ultralytics --name "Judo_Match_Clipper_vx.x.x" app.py
```
The executable is located in the `\dist` directory of the project folder.

---

## System Architecture & Development Guide

This section outlines the application's internal architecture, script interactions, and guidelines for future development and compilation.

### 1. High-Level Architecture Overview

The application is structured into three primary layers:
1. **Presentation Layer (`app.py`):** A PyQt/PySide-based graphical user interface that captures user configuration, handles standard output redirection (displaying terminal logs in the UI), and spawns the pipeline as an independent subprocess.
2. **Orchestration Layer (`end_to_end_pipeline.py`):** A Luigi-based task scheduler that manages the dependency graph of the video processing pipeline. It ensures tasks execute sequentially and handles state management.
3. **Execution Layer (`judo_footage_analysis/` & `scripts/`):** A collection of independent Python scripts that perform the actual I/O operations, video manipulation (FFmpeg), and machine learning inference (YOLOv8).

### 2. Pipeline Task Flow (Luigi Orchestrator)

The core pipeline (`end_to_end_pipeline.py`) consists of five sequential Luigi tasks. A downstream task will not initiate until its upstream dependencies emit a success state.

* **Task 1: Format Video:** Standardizes the raw input video (e.g., `.flv` to `.mp4`) to ensure consistent framerates and encoding for the AI model.
* **Task 2: Segment Videos (`truncate_videos.py`):** Utilizes `imageio_ffmpeg` and `ffmpeg.exe` to slice long tournament files into 10-minute (600-second) chunks to prevent memory overflow during inference.
* **Task 3: Generate Manifest (`generate_combat_json.py`):** Creates a structured JSON manifest tracking file paths and metadata for the ML model.
* **Task 4: Run AI Analysis (`extract_combat_phases.py`):** Ingests the manifest and runs YOLOv8 inference directly on the segmented videos to detect combat phases, outputting timestamps.
* **Task 5: Consolidate and Clip:** Reads the AI timestamp outputs and triggers FFmpeg to cut the final highlight clips losslessly.

### 3. Script Interaction & Dynamic Execution

To keep the ML repository (`judo_footage_analysis`) decoupled from the UI orchestrator, `end_to_end_pipeline.py` executes the execution layer scripts dynamically using `runpy.run_path()`. 

**Execution Flow Example (Task 4):**
1. Luigi initiates Task 4.
2. The orchestrator overrides `sys.argv` with the required CLI arguments (e.g., `--project-json`, `--output-dir`).
3. `sys.path.insert(0, os.path.abspath("."))` forces the Python environment to recognize the external folders.
4. `runpy.run_path("judo_footage_analysis/workflow/extract_combat_phases.py")` executes the script in the current memory space.
5. Upon successful exit code (0), Luigi writes a `Done` flag to the local cache and proceeds to the final clipping task.

### 4. Developer Guide

When modifying the pipeline or updating the application, adhere to the following constraints to prevent breaking the PyInstaller build.

#### A. Dependency Management & "Dummy Imports"
Because the orchestrator uses `runpy.run_path()` to execute external scripts dynamically, PyInstaller's static analyzer cannot see the third-party libraries used inside `judo_footage_analysis/` or `scripts/`. 

If you add a new library (e.g., `import pandas`) to any external script, you **must** add a corresponding "dummy import" to the top of `end_to_end_pipeline.py`:

```python
# end_to_end_pipeline.py
import pandas  # Required for PyInstaller to bundle the dependency
```

#### B. Multiprocessing Constraints (Windows Fork Bomb)
Windows lacks a native `fork()` command. Attempting to use Python's `multiprocessing` library to parallelize Luigi tasks or video segmentation will cause the compiled `.exe` to recursively spawn duplicate GUI windows, crashing the system.
* Ensure `multiprocessing.freeze_support()` remains immediately after `if __name__ == "__main__":` in `app.py`.
* Ensure all Luigi build commands within external scripts (e.g., `truncate_videos.py`) explicitly set `workers=1` and `local_scheduler=True`.

#### C. Compiling the Executable
When building a new release, you must instruct PyInstaller to collect the hidden data files (`.json`, `.txt`) associated with heavy libraries like PySpark and Ultralytics.

1. Delete existing `build/`, `dist/` directories, and the `.spec` file.
2. Run the compilation command:

```bash
pyinstaller --noconfirm --onedir --console --collect-all pyspark --collect-all ultralytics --name "Judo_AI_Clipper_vX.X.X" app.py
```
*(Note:`--windowed` may be used instead of `--console` to prevent a separate console window opening when using the GUI.*

3. After compilation, manually copy the following assets into the new `dist/Judo_Match_Clipper_vx.x.x/` directory:
    * `scripts/`
    * `judo_footage_analysis/`
    * `yolov8_custom.pt`
    * `ffmpeg.exe` and `ffprobe.exe` (Place inside `_internal/`)


---
# Original README and Note (Out of Date)

**The following section contains out of date information** regarding the function of the software but it is still recommended reading before building on top of the current work. This section includes the work of the original creators of the program and will help to understand the function and how we arrived at the current version of the tool.

### judo-footage-analysis

This repository is work supporting "Semi-Supervised Extraction and Analysis of Judo Combat Phases
from Recorded Live-Streamed Tournament Footage".
The goal of the project is to automate live-stream recording segmentation into individual matches, extract combat phases from matches, and to gather statistics at the tournament level.

This project was done as part of CS8813 Introduction to Research at Georgia Tech Europe during the Spring 2024 semester.

### quickstart

Checkout the repo and install any dependencies you may need to a virtual environment:

```bash
git checkout ${repo}
cd ${repo_name}

python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Install any of the relevant tools for running workflows:

- ffmpeg
- b2-tools
- google-cloud-sdk

#### running a workflow

Most of the data processing workflows are written as [luigi](https://github.com/spotify/luigi) scripts under the [judo_footage_analysis/workflow](./judo_footage_analysis/workflow) module.
These can be run as follows:

```bash
# in a terminal session
luigid

# in a separate session
python -m judo_footage_analysis.workflow.{module_name}
```

You can watch the progress of a job in the terminal or from the luigi web-ui at http://localhost:8082.


### Conversion of the Provided MKV File to Mp4

This project uses FFmpeg, provided through the Python Package
imageio-ffmpeg, bundles a local FFmpeg binary inside the virtual environment
This allows video processing without installing the python package on your OS


#### Activating the Virtual Environment
*If your virtual environment is already active skip this step*
```bash
.\.venv\Scripts\Activate
```
Once active, the prompt should look like:

`(.venv) PS C:\path\to\project`

#### Locating the FFmpeg Binary inside the venv

The following command prints the full path for FFmpeg executable to access it
```bash
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```
#### Converting MKV to MP4 Using the venv FFmpeg

Use the full FFmpeg path obtained above:

*Example:*
```
& "C:\Users\<username>\judo-footage-analysis-main\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe" -i "C:\Users\<username>\Desktop\OntarioOpen_Mat5_Sat2025.mkv" "C:\Users\<username>\Desktop\OntarioOpen_Mat5_Sat2025.mp4"
```
- `-i` in the command specifies the MKV file being inputted
- The last argument specifies the output MP4 file

Once you have the full FFmpeg path from Step 2, you can use it to run a conversion command. 
Because PowerShell cannot run executables with spaces in their path directly, we use `&` to invoke the executable.

### Setting the Output Folder for the Frames
This project allows extraction of frames from videos for further analysis. You can configure where these frames are saved, either using a default folder or specifying a custom location.
By default, frames are saved in folder relative to the input video "output_frames"

*You can specify a custom folder in your script or workflow. For example:*
```bash
from judo_footage_analysis.frame_extraction import extract_frames

video_path = "path/to/video.mp4"
output_folder = "path/to/output_frames"

extract_frames(video_path, output_folder)
```
- `output_folder` is the path where all extracted frames will be saved
- The folder will be automatically created if it does not exist

If your using the workflow, use the following command:
```
python -m judo_footage_analysis.workflow.extract_frames \
    --video "videos/match1.mp4" \
    --output_folder "frames/match1_frames"
```
This allows you to control where the output of the frames go.

### Automatic FLV to MP4 Conversion (FLV File Provided by Eugene)
The project includes a Python script to automatically convert FLV files to MP4. The script lives inside the repository, so you can run it directly from your project terminal

#### Running the Script
Open your terminal and turn on your virtual environment
```
.\.venv\Scripts\Activate
```
Call the conversion script
```
python scripts\convert_flv_auto.py
```
The script will do the following:
- Scan the folder specified inside the script (specified for Desktop) for .flv files
- Convert them to MP4
- Saves converted files in a `converted_mp4` folder in the same location

*NOTE: This script will only pick up your FLV Video File inside your Desktop and it should be outside the folder directly on the Desktop*

#### Video Segmentation of the Converted MP4 File
After converting to MP4, you can segment a long video into individual Judo matches using the `truncate_videos.py` workflow
```bash
python -m judo_footage_analysis.workflow.truncate_videos `
    --input-root-path "C:\Users\v5karthi\Desktop\converted_mp4" `
    --output-root-path "C:\Users\v5karthi\Desktop\segmented_matches" `
    --output-prefix "match_" `
    --duration 600 `
    --num-workers 1
```

What each variable means:
- `--input-root-path` – folder containing MP4 files to segment
- `--output-root-path` – folder where segmented matches will be saved

Output will be MP4 files in the specified output folder ready for frame extraction.

### Frame Extraction
Frames can be extracted from segmented videos for further analysis of each fight:
```bash
from judo_footage_analysis.frame_extraction import extract_frames

video_path = "path/to/match.mp4"
output_folder = "path/to/frames"

extract_frames(video_path, output_folder)
```

or you can use the workflow:
```bash
python -m judo_footage_analysis.workflow.extract_frames \
    --video "videos/match1.mp4" \
    --output_folder "frames/match1_frames"
```
### Generating the Project JSON (Required for Later Workflows)
Some workflows in this repository require a JSON file listing all videos to be processed.
To simplify this step, a script is included to automatically create this JSON file.

A script named `generate_video_json.py` is located under the `scripts/` folder. It scans a folder containing your MP4 files and generates a JSON listing each video.
*Virtual Envoirnment should be active at all times throughout this code unless specified*

**Run the JSON generator:**
```bash
python scripts\generate_video_json.py
```
**This script creates a JSON file at:**
```bash
judo-footage-analysis-main/data/combat_phase/project.json
```
### Combat Phase Extraction
#### Install dependencies

```bash
pip install -r requirements.txt
pip install imageio[ffmpeg]
````

After preprocessing, run the phase classifier:
```bash
python -m judo_footage_analysis.workflow.extract_combat_phases \
    --project-json data/combat_phase/project.json \
    --output-dir data/combat_phase/results
```
This workflow does the following:
- Frame loading
- Pose or motion feature extraction (depending on model)
- Semi-supervised classification of combat phases

### Video Segmentation
After converting your livestream recording to MP4, you can segment the long file into individual Judo matches using the `truncate_videos` Luigi workflow.

This workflow cuts the video into fixed-length segments (10 minutes each) and saves them to an output folder.

*NOTE: Segmenting very large recordings (20–30 GB+) can take several hours, especially on laptops. Segments will appear one by one in your `segmented_matches/` folder as they finish.*

**Running the Video Segmentation Workflow**
In your activated virtual environment
```bash
python -m judo_footage_analysis.workflow.truncate_videos \
    --input-root-path "C:\Users\<username>\Desktop\converted_mp4" \
    --output-root-path "C:\Users\<username>\Desktop\segmented_matches" \
    --output-prefix "match_" \
    --duration 600 \
    --num-workers 1
```
What it means:
- `--input-root-path` - Folder containing the input MP4 file to segment
- `--output-root-path` - Folder where segmented match clips will be saved
- `--output-prefix` - Prefix applied to each segment file name
- `--duration` - Length (in seconds) of each output clip (example: `600` = 10 minutes)
- `--num-workers` - Number of parallel workers; keep at 1 on most laptops

  **Example Output Files**
```bash
match_0001.mp4
match_0002.mp4
match_0003.mp4
```
They will keep on appearing until the segmenting is complete. If you want longer commands edit the `--duration` variable to edit the time.
Segments will appear one by one in your `segmented_matches/` folder as they finish and with it finishing it'll issue a _SUCCESS file which'll confirm that the segmenting is done.


### Combat Phase Extraction (Machine Learning)

This workflow uses a YOLOv8 object detection model to analyze judo matches and classify combat into Tachi-waza (standing) or Ne-waza (groundwork) based on athlete bounding box statistics.

**Generating the Project JSON**

Before running the ML workflow, you must generate a project manifest. This script scans your segmented matches and creates a "map" for the AI.

```bash
# Run the JSON generator
python scripts/generate_combat_json.py
```
- Input Folder: Scans `Desktop/segmented_matches` by default
- Output File: Saves the manifest to `data/combat_phase/project.json`

**Running the Extraction Workflow**
Ensure the `luigid` scheduler is running in a separate terminal window. Then execute the extraction using the following commands:

```bash
# Set PYTHONPATH so Python recognizes the local project modules
$env:PYTHONPATH = "."

# Run the ML Workflow Task
python -m judo_footage_analysis.workflow.extract_combat_phases ExtractCombatPhases `
    --project-json "data/combat_phase/project.json" `
    --output-dir "data/combat_phase/results"
```


*Key Features:*
- Automatically downloads the yolov8n.pt weights on the first run
- Includes a built-in fix for Windows `CERTIFICATE_VERIFY_FAILED` errors during model downloads
- Handles various `JSON` keys (e.g., `video`, `path`, `video_path`) to prevent `KeyError` crashes

**Data Consolidation**
The final step generates the research metrics.
```bash
python scripts/analyze_and_visualize.py --input-file "data/tournament_master_log.csv" --output-file "data/intensity_mapped_results.csv"
```

*The Output*

`intensity_mapped_results.png`
- This chat provides visual map of the match
      - Green Regions: Tachi-waza (Standing combat)
      - Blue Regions: Ne-waza (Groundwork/Grappling)
      - Red Regions: Mate (Intermission/Stoppages)
      - Black Line: Smoothed Intensity (The velocity of athlete movement)
      - "MAJOR ACTION" Markers: Automatic labels identifying explosive spikes (likely throws)

   `intensity_mapped_results.csv`
  - The raw dataset for statistical research including
      - `smoothed_intensity`: Velocity-based activity levels
      - `phase`: The classified state of the match at that timestamp
      - `detections`: Raw YOLOv8 coordinate strings for custom spatial analysis

### Technical Maintenance

*Smart Logic*
The tracking focuses on the primary athletes by selecting the largest bounding box area in each frame. If the camera angle changes and background figures (like referees) appear larger than the athletes, you may need to add a class-based filter in `scripts/analyze_and_visualize.py`

*Coordinate Scaling*
The analysis script includes a "Boost" feature. If the YOLOv8 model outputs normalized coordinates (values between 0 and 1), the script automatically scales them by 1000 to ensure intensity peaks are visible on the Y-axis.

### Core Dependencies Used
- `ultralytics` (YOLOv8 Engine)
- `luigi` (Workflow Management)
- `pandas & numpy` (Data Analytics)
- `seaborn & matplotlib` (Visualization)
- `opencv-python` (Frame Processing)
