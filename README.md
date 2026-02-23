# AI Personal Trainer & Workout Assistant 🏋️‍♂️🤖

An advanced, AI-powered workout assistant that tracks exercises, counts repetitions, and provides real-time biomechanical feedback on your form. Built with Python, TensorFlow, OpenCV, and MediaPipe.

## 🌟 Features
* **Dual-Inference AI Architecture**: Combines an MLP pose classifier with an LSTM-based motion sequence model.
* **Form Evaluation**: Real-time feedback on your posture (e.g., "Keep your back straight", "Go lower").
* **Repetition Counting**: Robust counting using a custom Finite State Machine (FSM).
* **Modern GUI**: Built with `customtkinter` for an intuitive user experience.
* **Live & File Modes**: Analyze pre-recorded videos or use your webcam for live sessions.
* **Automated JSON Reports**: Generates detailed workout summaries including scores and specific coach feedback.

## 🏗️ System Architecture
This project implements a sophisticated pipeline to ensure accurate exercise recognition and counting:
1. **Pose MLP Model**: Analyzes single frames to classify static poses (e.g., "squat_down", "pushup_up").
2. **Sequential LSTM Model**: Extracts temporal meta-features over a sliding window (30 frames) to capture motion dynamics.
3. **Hybrid OOF Model**: Merges the probabilities from the Pose Model with the embeddings from the Sequence Model for highly robust classification.
4. **Finite State Machine (FSM)**: Eliminates false positives in repetition counting by enforcing strict state transitions (START -> MID -> START).

## 📁 Directory Structure
```text
ai_personal_trainer/
├── project_code/               # Training Pipeline & EDA
│   ├── app/                        # Main Application Code (GUI & Inference)
│       ├── app.py                  # CustomTkinter GUI Dashboard
│       ├── camera_input_checking.py# Camera diagnostics
│       └── report_with_grade.py    # Core backend pipeline
│   ├── combined_model/         # Hybrid model training scripts
│   ├── pose_model/             # MLP pose model training
│   ├── preparing_datasets/     # Data cleaning, merging, and OOF generation
│   └── sequence_model/         # LSTM sequence model training
├── datasets/                   # (Ignored) Raw & Processed Datasets
├── model/                      # (Ignored) Saved .keras models and scalers
└── output/                     # (Ignored) Reports, plots, and recordings
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/szymonbloch/ai-personal-trainer.git
   cd ai-personal-trainer
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights:**
   Because the `.keras` and `.pkl` files are too large for standard GitHub hosting, they are ignored in this repository. 
   * Download the `model.zip` file from the **[Releases]** tab of this repository.
   * Extract the contents directly into the root directory so that the `model/` folder structure matches the layout expected by the code.

## 🎮 Usage

Launch the main dashboard by running:
```bash
python project_code/app/app.py
```

From the GUI, you can:
* **📂 Load Video File**: Select a pre-recorded workout and click "Analyze Video".
* **🎥 Record Training**: Record your workout locally via webcam.
* **🔴 Start LIVE Session**: Perform exercises directly in front of the camera with real-time feedback.
* **📄 Open Report JSON**: View and inspect detailed insights from past sessions.

## 🧠 Technologies Used
* **Deep Learning**: TensorFlow, Keras
* **Computer Vision**: OpenCV, MediaPipe Pose
* **Machine Learning**: Scikit-Learn, Pandas, NumPy
* **Interface**: CustomTkinter
