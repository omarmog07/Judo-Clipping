import sys
import os
import shutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject

# Import the pipeline natively so PyInstaller bundles it
import luigi
import end_to_end_pipeline 
import multiprocessing

# ==========================================
# 1. STREAM ROUTER (Terminal to GUI)
# ==========================================
class EmittingStream(QObject):
    """Catches Python print() statements and converts them to PyQt signals."""
    textWritten = pyqtSignal(str)
    
    def write(self, text):
        if text.strip(): 
            self.textWritten.emit(str(text))
            
    def flush(self): 
        pass

# ==========================================
# 2. BACKGROUND THREAD (Prevents UI Freeze)
# ==========================================
class PipelineWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, video_paths, base_dir):
        super().__init__()
        self.video_paths = video_paths # Now strictly accepts a list
        self.base_dir = base_dir

    def run(self):
        # 1. Generate the folder structure
        raw_dir = os.path.join(self.base_dir, "01_Raw_Input")
        folders = [raw_dir, "02_Converted", "03_Segmented", "04_Frames", "05_Results", "06_Final_Clips"]
        for f in folders:
            os.makedirs(os.path.join(self.base_dir, f), exist_ok=True)

        self.progress_signal.emit(f"Created folder structure at {self.base_dir}")

        # 2. Handle the dropped files (Loop through the array)
        self.progress_signal.emit(f"Copying {len(self.video_paths)} video(s) to raw folder...")
        for vp in self.video_paths:
            dest_path = os.path.join(raw_dir, os.path.basename(vp))
            shutil.copy2(vp, dest_path)
            self.progress_signal.emit(f"Successfully copied {os.path.basename(vp)}")

        # 3. Override standard terminal output to route to the GUI
        sys.stdout = EmittingStream()
        sys.stdout.textWritten.connect(self.emit_log)
        sys.stderr = sys.stdout # Catch error tracebacks too
        
        print("\nLaunching Judo AI Pipeline natively...")
        
        # 4. Run Luigi directly inside the current process memory (Sequential Processing)
        luigi.build([end_to_end_pipeline.Task6_ConsolidateAndClip()], workers=1, local_scheduler=True)
        
        print("PIPELINE COMPLETE!")
        
        # 5. Restore normal terminal output
        sys.stdout = sys.__stdout__ 
        sys.stderr = sys.__stderr__
        
        self.finished_signal.emit()

    def emit_log(self, text):
        self.progress_signal.emit(text)

# ==========================================
# 3. MAIN USER INTERFACE
# ==========================================
class JudoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Match Clipper v1.1.9")
        self.resize(650, 500) # Increased height slightly to fit text
        self.setAcceptDrops(True) 

        layout = QVBoxLayout()
        
        # --- NEW LOGIC: Instruction Text ---
        self.instructions = QLabel(
            "<b>Instructions:</b><br>"
            "1. Drag and drop raw Judo video files (.mp4, .flv) into the dashed box below.<br>"
            "2. The system will automatically build the working directory and process the AI inferences. A terminal window may open, do not close it.<br>"
            "3. Final segmented matches will be saved in the <b>Judo_Pipeline\\06_Final_Clips</b> folder.<br>"
	    "<br>"
	    "This program may take some time to run. For stability, avoid running other programs while processing.<br>"
        )
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("font-size: 14px; padding-bottom: 10px;")
        layout.addWidget(self.instructions)
        # -----------------------------------
        
        self.drop_label = QLabel("Drag & Drop Judo Match Video(s) Here (.mp4, .flv)")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("border: 2px dashed #aaa; font-size: 14px; padding: 50px;")
        layout.addWidget(self.drop_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #d9d9d9; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.console)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # Grabs the paths of ALL dropped files
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.start_pipeline(files)

    def start_pipeline(self, video_paths):
        # Safely extract the length of the list for the UI
        self.drop_label.setText(f"Processing {len(video_paths)} file(s)...")
        self.progress_bar.setRange(0, 0) # Infinite loading animation
        
        base_pipeline_dir = os.path.abspath("Judo_Pipeline")
        
        self.worker = PipelineWorker(video_paths, base_pipeline_dir)
        self.worker.progress_signal.connect(self.update_console)
        self.worker.finished_signal.connect(self.pipeline_finished)
        self.worker.start()

    def update_console(self, text):
        self.console.append(text)

    def pipeline_finished(self):
        self.progress_bar.setRange(0, 1) 
        self.progress_bar.setValue(1) 
        self.drop_label.setText("Drag & Drop Judo Match Video(s) Here (.mp4, .flv)")

# ==========================================
# 4. TRIGGER
# ==========================================
if __name__ == "__main__":

    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = JudoApp()
    window.show()
    sys.exit(app.exec())
