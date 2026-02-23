import sys
import cv2
import json
import time
import subprocess
import customtkinter as ctk
from PIL import Image
from pathlib import Path
from tkinter import filedialog
from typing import Tuple, Dict, Any, Optional

# Appearance Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    """Main GUI Application for the AI Personal Trainer."""

    def __init__(self):
        super().__init__()

        self.title("AI Personal Trainer - Dashboard")
        self.geometry("1250x750")

        # System paths configuration
        self.base_dir = Path(__file__).resolve().parent.parent
        self.app_dir = self.base_dir / "app"
        self.output_dir = self.base_dir.parent / "output" / "analysis_reports"
        self.recordings_dir = self.base_dir.parent / "output" / "recordings"

        self.backend_script = self.app_dir / "report_with_grade.py"

        # State variables
        self.video_source = 1
        self.is_recording = False
        self.vid_capture = None
        self.vid_writer = None
        self.current_video_path: Optional[str] = None

        self._build_ui()

    # =========================================================
    # UI CONSTRUCTION
    # =========================================================

    def _build_ui(self) -> None:
        """Constructs the main grid layout and instantiates UI components."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self) -> None:
        """Constructs the left sidebar with navigation buttons."""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.label_menu = ctk.CTkLabel(
            self.sidebar, text="TRAINER MENU", font=ctk.CTkFont(size=20, weight="bold")
        )
        self.label_menu.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Menu Buttons
        self.btn_upload = ctk.CTkButton(
            self.sidebar, text="📂 Load Video File", command=self.upload_video
        )
        self.btn_upload.grid(row=1, column=0, padx=20, pady=20)

        self.btn_record = ctk.CTkButton(
            self.sidebar, text="🎥 Record Training", fg_color="green", command=self.toggle_recording
        )
        self.btn_record.grid(row=2, column=0, padx=20, pady=20)

        self.btn_analyze = ctk.CTkButton(
            self.sidebar, text="⚙️ Analyze Video", command=self.analyze_video
        )
        self.btn_analyze.grid(row=3, column=0, padx=20, pady=20)

        self.btn_live = ctk.CTkButton(
            self.sidebar, text="🔴 Start LIVE Session", fg_color="#b30000",
            hover_color="#800000", command=self.run_live_analysis
        )
        self.btn_live.grid(row=4, column=0, padx=20, pady=20)

        self.btn_load_json = ctk.CTkButton(
            self.sidebar, text="📄 Open Report JSON", fg_color="gray", command=self.load_existing_report
        )
        self.btn_load_json.grid(row=5, column=0, padx=20, pady=20)

    def _build_main_area(self) -> None:
        """Constructs the main area containing the video preview and report details."""
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.main_area.grid_columnconfigure(0, weight=3)
        self.main_area.grid_columnconfigure(1, weight=2)
        self.main_area.grid_rowconfigure(0, weight=1)

        # Video Container
        self.video_frame = ctk.CTkFrame(self.main_area, corner_radius=10)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.camera_label = ctk.CTkLabel(
            self.video_frame,
            text="Video Preview Area\n\nLoad a video or start recording",
            font=ctk.CTkFont(size=16)
        )
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        # Results Container
        self.results_frame = ctk.CTkFrame(self.main_area, corner_radius=10)
        self.results_frame.grid(row=0, column=1, sticky="nsew")

        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.label_results = ctk.CTkLabel(
            self.results_frame, text="📊 TRAINING REPORT", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.label_results.grid(row=0, column=0, sticky="w", padx=20, pady=15)

        self.results_scrollable = ctk.CTkScrollableFrame(self.results_frame, label_text="Details")
        self.results_scrollable.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    # =========================================================
    # VALIDATION FUNCTIONS
    # =========================================================

    def validate_video_file(self, filepath: str) -> Tuple[bool, str]:
        """Checks if the video file is valid and readable by OpenCV."""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False, "Corrupted header or unsupported codec."

            ret, _ = cap.read()
            cap.release()

            if not ret:
                return False, "File is empty or corrupted (0 frames read)."

            return True, "OK"
        except Exception as e:
            return False, f"Read error: {str(e)}"

    # =========================================================
    # RENDER FUNCTIONS
    # =========================================================

    def show_status_message(self, message: str, is_error: bool = False) -> None:
        """Displays status or error messages in the results panel."""
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()

        color = "#ff5555" if is_error else "#ffffff"
        icon = "❌" if is_error else "ℹ️"

        lbl = ctk.CTkLabel(
            self.results_scrollable, text=f"{icon} STATUS",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=color
        )
        lbl.pack(pady=(20, 10))

        lbl_msg = ctk.CTkLabel(
            self.results_scrollable, text=message, wraplength=300,
            justify="center", font=ctk.CTkFont(size=14)
        )
        lbl_msg.pack(pady=10)

    def create_stat_card(self, parent: ctk.CTkFrame, label: str, value: Any, highlight: bool = False) -> None:
        """Creates a UI card displaying a key statistic."""
        card = ctk.CTkFrame(parent, fg_color=("gray85", "gray25"))
        card.pack(fill="x", pady=5, padx=5)

        lbl_title = ctk.CTkLabel(
            card, text=label.upper(), font=ctk.CTkFont(size=12, weight="bold"), text_color="gray"
        )
        lbl_title.pack(anchor="w", padx=10, pady=(5, 0))

        val_color = "#4ade80" if highlight else "white"
        lbl_val = ctk.CTkLabel(
            card, text=str(value), font=ctk.CTkFont(size=20, weight="bold"), text_color=val_color
        )
        lbl_val.pack(anchor="w", padx=10, pady=(0, 5))

    def create_progress_bar_widget(self, parent: ctk.CTkFrame, key: str, value: Any) -> None:
        """Renders a progress bar for ratings or percentage scores."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=10, padx=5)

        try:
            val_float = float(value)
            is_rating = any(x in key.lower() for x in ['rating', 'grade', 'ocena'])

            if is_rating and val_float <= 6.0:
                display_text = f"{key.upper()}: {int(val_float)}/5"
                normalized_val = val_float / 5.0
            else:
                display_text = f"{key.upper()}: {value}"
                normalized_val = val_float if val_float <= 1.0 else val_float / 100.0
        except ValueError:
            return

        header = ctk.CTkLabel(frame, text=display_text, font=ctk.CTkFont(size=14, weight="bold"))
        header.pack(anchor="w", padx=5)

        progress = ctk.CTkProgressBar(frame, height=15)
        progress.set(normalized_val)

        if normalized_val >= 0.8:
            progress.configure(progress_color="#2ecc71")
        elif normalized_val >= 0.5:
            progress.configure(progress_color="#f1c40f")
        else:
            progress.configure(progress_color="#e74c3c")

        progress.pack(fill="x", padx=5, pady=5)

    def render_feedback_list(self, parent: ctk.CTkFrame, feedback_data: Any) -> None:
        """Renders a bulleted list of coach feedback or errors."""
        fb_frame = ctk.CTkFrame(parent, fg_color=("gray90", "gray20"))
        fb_frame.pack(fill="x", pady=15, padx=5)

        lbl_title = ctk.CTkLabel(
            fb_frame, text="COACH FEEDBACK", font=ctk.CTkFont(size=14, weight="bold"), text_color="gray"
        )
        lbl_title.pack(anchor="w", padx=10, pady=(10, 5))

        items = feedback_data if isinstance(feedback_data, list) else ([str(feedback_data)] if feedback_data else [])

        if not items:
            lbl = ctk.CTkLabel(
                fb_frame, text="✅ No errors detected.",
                text_color="#2ecc71", font=ctk.CTkFont(size=14, weight="bold")
            )
            lbl.pack(pady=(5, 15), anchor="w", padx=10)
            return

        lbl_warn = ctk.CTkLabel(
            fb_frame, text="⚠️ Corrections needed:", text_color="#e67e22", font=ctk.CTkFont(weight="bold")
        )
        lbl_warn.pack(pady=(0, 5), anchor="w", padx=10)

        for item in items:
            row = ctk.CTkFrame(fb_frame, fg_color="transparent")
            row.pack(fill="x", pady=2, padx=10)

            ctk.CTkLabel(
                row, text="•", font=ctk.CTkFont(size=20), width=20, text_color="#ff5555"
            ).pack(side="left", anchor="n")

            ctk.CTkLabel(
                row, text=str(item), justify="left", anchor="w",
                wraplength=280, text_color=("black", "white")
            ).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(fb_frame, text="", height=5).pack()

    def render_json_recursive(self, parent: ctk.CTkFrame, data: Dict[str, Any]) -> None:
        """Parses and renders JSON report data into UI components."""
        if not isinstance(data, dict):
            return

        if 'status' in data:
            status_val = data['status']
            status_color = "#2ecc71" if status_val == 'ok' else "#e74c3c"
            st_frame = ctk.CTkFrame(parent, fg_color=status_color, height=35)
            st_frame.pack(fill="x", pady=(0, 10))
            ctk.CTkLabel(
                st_frame, text=f"STATUS: {status_val.upper()}",
                text_color="white", font=ctk.CTkFont(weight="bold")
            ).pack(pady=5)

        stats_frame = ctk.CTkFrame(parent, fg_color="transparent")
        stats_frame.pack(fill="x")

        if data.get('exercise'):
            self.create_stat_card(
                stats_frame, "Exercise", str(data['exercise']).replace('_', ' ').upper(), highlight=True
            )
        if 'reps' in data:
            self.create_stat_card(stats_frame, "Repetitions", data['reps'])

        for key in ['rating', 'score', 'ocena']:
            if key in data:
                self.create_progress_bar_widget(parent, key, data[key])

        if 'feedback' in data:
            self.render_feedback_list(parent, data['feedback'])
        elif data.get('status') == 'ok':
            self.render_feedback_list(parent, [])

    # =========================================================
    # APP LOGIC & EVENT HANDLERS
    # =========================================================

    def upload_video(self) -> None:
        """Handles video file selection and validation."""
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not filename:
            return

        is_valid, error_msg = self.validate_video_file(filename)

        if not is_valid:
            self.current_video_path = None
            self.show_status_message(
                f"❌ INVALID VIDEO FILE:\n{Path(filename).name}\n\nReason: {error_msg}", is_error=True
            )
            self.camera_label.configure(image=None, text="Invalid file selected.")
            return

        self.current_video_path = filename

        # Render first frame as thumbnail
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 480))
            img_tk = ctk.CTkImage(light_image=Image.fromarray(frame), size=(640, 480))
            self.camera_label.configure(image=img_tk, text="")
        cap.release()

        self.show_status_message(
            f"✅ File loaded successfully:\n{Path(filename).name}\n\nClick 'Analyze Video' to start."
        )

    def toggle_recording(self) -> None:
        """Starts or stops local video recording from the camera."""
        if not self.is_recording:
            selected_dir = filedialog.askdirectory(
                initialdir=str(self.recordings_dir),
                title="Wybierz folder do zapisu nagrania"
            )
            if not selected_dir:
                return

            self.is_recording = True
            self.btn_record.configure(text="⏹ Stop Recording", fg_color="red")
            self.vid_capture = cv2.VideoCapture(self.video_source)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = int(time.time())

            self.current_video_path = str(Path(selected_dir) / f'training_{timestamp}.avi')
            self.vid_writer = cv2.VideoWriter(self.current_video_path, fourcc, 20.0, (640, 480))

            self.update_camera()
        else:
            self.is_recording = False
            self.btn_record.configure(text="🎥 Record Training", fg_color="green")

            if self.vid_capture:
                self.vid_capture.release()
            if self.vid_writer:
                self.vid_writer.release()

            self.camera_label.configure(image=None, text="Video saved successfully.")
            self.show_status_message(f"Video saved at:\n{self.current_video_path}\n\nReady to analyze.")

    def update_camera(self) -> None:
        """Continuously reads frames from the camera and updates the UI."""
        if self.is_recording and self.vid_capture.isOpened():
            ret, frame = self.vid_capture.read()
            if ret:
                self.vid_writer.write(frame)
                img_tk = ctk.CTkImage(
                    light_image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    size=(640, 480)
                )
                self.camera_label.configure(image=img_tk, text="")
                self.camera_label.image = img_tk
                self.after(20, self.update_camera)

    def _execute_analysis_backend(self, mode: str, output_json_path: str, camera_idx: str = "0") -> None:
        """Helper function to execute the backend analysis script."""
        command = [
            sys.executable, str(self.backend_script),
            "--mode", mode,
            "--out", output_json_path
        ]

        if mode == "file" and self.current_video_path:
            command.extend(["--path", self.current_video_path])
        elif mode == "live":
            command.extend(["--camera_index", camera_idx])

        try:
            subprocess.run(command, capture_output=(mode == "file"), text=True, check=True)
            if Path(output_json_path).exists():
                self.display_json_file(output_json_path)
            else:
                self.show_status_message("Report not generated.", is_error=True)

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr if mode == "file" else "Live session crashed."
            self.show_status_message(f"Script Error:\n{err_msg}", is_error=True)
            print(e.stderr if mode == "file" else e)
        except Exception as e:
            self.show_status_message(f"Unexpected Error:\n{str(e)}", is_error=True)

    def analyze_video(self) -> None:
        """Triggers the backend analysis on a selected video file."""
        if not self.current_video_path:
            self.show_status_message("Error: Load a video first!", is_error=True)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        default_filename = f"report_{int(time.time())}.json"

        output_json_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON file", "*.json")],
            initialdir=str(self.output_dir),
            initialfile=default_filename,
            title="Save Report Location"
        )

        if not output_json_path:
            return

        self.show_status_message("Running AI Analysis...\nThis may take a moment.")
        self.update()
        self._execute_analysis_backend(mode="file", output_json_path=output_json_path)

    def run_live_analysis(self) -> None:
        """Triggers the backend analysis in live camera mode."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        default_filename = f"live_report_{int(time.time())}.json"

        output_json_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON file", "*.json")],
            initialdir=str(self.output_dir),
            initialfile=default_filename,
            title="Save Live Report Location"
        )

        if not output_json_path:
            return

        self.show_status_message("Launching LIVE Window...\nPress 'q' in the camera window to finish.")
        self.update()
        self._execute_analysis_backend(mode="live", output_json_path=output_json_path)

    def load_existing_report(self) -> None:
        """Opens a file dialog to load an existing JSON report."""
        filename = filedialog.askopenfilename(
            initialdir=str(self.output_dir),
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            self.display_json_file(filename)

    def display_json_file(self, filepath: str) -> None:
        """Loads and triggers rendering of a JSON report."""
        if not Path(filepath).exists():
            self.show_status_message("Report file not found!", is_error=True)
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for widget in self.results_scrollable.winfo_children():
                widget.destroy()

            self.render_json_recursive(self.results_scrollable, data)
        except Exception as e:
            self.show_status_message(f"Corrupted JSON file:\n{e}", is_error=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()