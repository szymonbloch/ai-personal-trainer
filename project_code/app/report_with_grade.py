import os
import cv2
import json
import joblib
import argparse
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from collections import deque, Counter
from typing import List, Dict, Tuple, Any, Optional

from tensorflow.keras.models import load_model

# =========================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================

SEQ_LEN = 30
STEP = 2
MIN_VISIBILITY_THRESHOLD = 0.6
MIN_FRAMES_FOR_VALID_EXERCISE = 10


def get_system_paths(base_dir: Path) -> Dict[str, Path]:
    """Generates all required paths for models and scalers dynamically."""
    return {
        "count_model": base_dir / "model" / "pose_model_MLP" / "pose_model_pose_dataset.keras",
        "count_scaler": base_dir / "model" / "pose_model_MLP" / "pose_model_scaler_pose_dataset.pkl",
        "count_enc": base_dir / "model" / "pose_model_MLP" / "pose_model_encoder_pose_dataset.pkl",

        "hybrid_feeder_model": base_dir / "model" / "pose_model_MLP" / "pose_model_seq_dataset.keras",
        "hybrid_feeder_scaler": base_dir / "model" / "pose_model_MLP" / "pose_model_scaler_seq_dataset.pkl",

        "old_seq_model": base_dir / "model" / "sequence_model_LSTM" / "sequence_model.keras",
        "old_seq_scaler": base_dir / "model" / "sequence_model_LSTM" / "sequence_scaler.pkl",
        "feature_cols": base_dir / "datasets" / "ready_to_train" / "sequence_feature_cols.json",

        "hybrid_model": base_dir / "model" / "combined_model" / "combined_sequence_model.keras",
        "ex_enc": base_dir / "model" / "combined_model" / "sequence_encoder_from_probs.pkl"
    }


# =========================================================
# 2. HELPER CLASSES (FSM, Evaluator, Abstainer)
# =========================================================


class BinaryFSM:
    """Finite State Machine for counting exercise repetitions."""

    def __init__(self, start_state_name: str, mid_state_name: str, prob_th: float = 0.60, hold_frames: int = 3):
        self.start_name = start_state_name
        self.mid_name = mid_state_name
        self.prob_th = prob_th
        self.hold_frames = hold_frames

        self.state = "START"
        self.counter = 0
        self.frames_in_zone = 0

    def update(self, phase_probs: np.ndarray, phase_idx_map: Dict[str, int]) -> int:
        if self.start_name not in phase_idx_map or self.mid_name not in phase_idx_map:
            return 0

        p_start = phase_probs[phase_idx_map[self.start_name]]
        p_mid = phase_probs[phase_idx_map[self.mid_name]]

        if self.state == "START":
            if p_mid > self.prob_th:
                self.frames_in_zone += 1
                if self.frames_in_zone >= self.hold_frames:
                    self.state = "MID"
                    self.frames_in_zone = 0
            else:
                self.frames_in_zone = 0

        elif self.state == "MID":
            if p_start > self.prob_th:
                self.frames_in_zone += 1
                if self.frames_in_zone >= self.hold_frames:
                    self.state = "START"
                    self.counter += 1
                    self.frames_in_zone = 0
            else:
                self.frames_in_zone = 0

        return self.counter


class ExerciseEvaluator:
    """Evaluates biomechanical form based on 3D landmarks."""

    @staticmethod
    def _get_coords(landmarks: Any, idx: int) -> np.ndarray:
        return np.array([landmarks[idx].x, landmarks[idx].y])

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        return 360 - angle if angle > 180.0 else angle

    def evaluate(self, landmarks: Any, exercise_name: str) -> List[str]:
        lm = landmarks
        errors = []

        if exercise_name == "squat":
            hip, knee, ankle = self._get_coords(lm, 24), self._get_coords(lm, 26), self._get_coords(lm, 28)
            if self._calculate_angle(hip, knee, ankle) < 120:
                hip_y = (lm[23].y + lm[24].y) / 2
                knee_y = (lm[25].y + lm[26].y) / 2

                if hip_y < (knee_y - 0.05):
                    errors.append("Go lower (hips too high)")

                if abs(lm[25].x - lm[26].x) < abs(lm[27].x - lm[28].x) * 0.80:
                    errors.append("Watch your knees (caving in)")

        elif exercise_name == "push_up":
            shoulder, hip, ankle = self._get_coords(lm, 11), self._get_coords(lm, 23), self._get_coords(lm, 27)
            if self._calculate_angle(shoulder, hip, ankle) < 160:
                errors.append("Keep your back straight")

        elif exercise_name == "jumping_jack":
            if abs(lm[15].x - lm[16].x) > 2.5 * abs(lm[11].x - lm[12].x):
                if ((lm[15].y + lm[16].y) / 2) > lm[0].y:
                    errors.append("Arms higher!")

        return errors


class SmoothAbstain:
    """Smooths out predictions over time and abstains if confidence is low."""

    def __init__(self, n_classes: int, smooth_k: int = 5, conf_th: float = 0.60):
        self.buf = deque(maxlen=smooth_k)
        self.conf_th = conf_th

    def update_and_decide(self, probs: np.ndarray) -> Any:
        self.buf.append(probs)
        avg = np.mean(self.buf, axis=0)
        idx = np.argmax(avg)
        return "unknown" if avg[idx] < self.conf_th else idx


# =========================================================
# 3. MATH & FEATURE EXTRACTION
# =========================================================

def calculate_angle_3d(a: List[float], b: List[float], c: List[float]) -> float:
    a_arr, b_arr, c_arr = np.array(a), np.array(b), np.array(c)
    ba = a_arr - b_arr
    bc = c_arr - b_arr
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def preprocess_pose_99(landmarks: Any, scaler: Any) -> np.ndarray:
    """Extracts 99 spatial coordinates, centers by hip, scales by height."""
    x = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    x -= x[24].copy()
    h = np.linalg.norm(x[0] - x[32])
    x /= (h if h > 1e-6 else 1.0)
    return scaler.transform(x.reshape(1, -1))


def extract_features_170(results: Any, feature_cols: List[str]) -> Optional[np.ndarray]:
    """Extracts engineered biomechanical features for the motion model."""
    if not results.pose_world_landmarks:
        return None

    lm = results.pose_world_landmarks.landmark
    d = {}

    # Feature mapping logic (kept mostly identical but structured cleanly)
    names = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
             "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_pinky_1", "right_pinky_1", "left_index_1",
             "right_index_1", "left_thumb_2", "right_thumb_2", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]

    for i, n in enumerate(names):
        d[f"x_{n}"], d[f"y_{n}"], d[f"z_{n}"] = lm[i].x * 100, lm[i].y * 100, lm[i].z * 100

    coords = lambda i: [lm[i].x, lm[i].y, lm[i].z]

    d["right_elbow_right_shoulder_right_hip"] = calculate_angle_3d(coords(14), coords(12), coords(24))
    d["left_elbow_left_shoulder_left_hip"] = calculate_angle_3d(coords(13), coords(11), coords(23))

    mid_hip = [(lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2, (lm[23].z + lm[24].z) / 2]
    d["right_knee_mid_hip_left_knee"] = calculate_angle_3d(coords(26), mid_hip, coords(25))

    d["right_hip_right_knee_right_ankle"] = calculate_angle_3d(coords(24), coords(26), coords(28))
    d["left_hip_left_knee_left_ankle"] = calculate_angle_3d(coords(23), coords(25), coords(27))
    d["right_wrist_right_elbow_right_shoulder"] = calculate_angle_3d(coords(16), coords(14), coords(12))
    d["left_wrist_left_elbow_left_shoulder"] = calculate_angle_3d(coords(15), coords(13), coords(11))

    try:
        ordered = [d.get(c, 0.0) for c in feature_cols]
        return np.array(ordered, dtype=np.float32)
    except KeyError as e:
        print(f"[ERROR] Missing feature column mapping: {e}")
        return None


# =========================================================
# 4. DRAWING & HUD UTILITIES
# =========================================================

def draw_no_person_warning(frame: np.ndarray) -> None:
    """Draws a warning overlay when no person is detected."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    text = "NO PERSON DETECTED"
    font_scale = 1.2
    thickness = 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)


def draw_workout_hud(frame: np.ndarray, exercise: str, reps: int, feedback: str) -> None:
    """Draws the Heads-Up Display with current workout metrics."""
    cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)

    cv2.putText(frame, f"EXERCISE: {exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"REPS: {reps}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    color_fb = (0, 255, 0) if feedback == "Good form" else (0, 0, 255)
    cv2.putText(frame, f"HINT: {feedback}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fb, 2)


# =========================================================
# 5. CORE PIPELINE
# =========================================================

def init_fsm_machines(phase_idx: Dict[str, int]) -> Dict[str, BinaryFSM]:
    """Initializes Finite State Machines for supported exercises."""
    configs = [
        ('squat', 'squats_up', 'squats_down'),
        ('push_up', 'pushups_up', 'pushups_down'),
        ('jumping_jack', 'jumping_jacks_down', 'jumping_jacks_up'),
        ('pull_up', 'pullups_down', 'pullups_up'),
        ('situp', 'situp_down', 'situp_up')
    ]
    fsms = {}
    for ex, start, mid in configs:
        s = start if start in phase_idx else f"{start}s"
        m = mid if mid in phase_idx else f"{mid}s"
        if s in phase_idx and m in phase_idx:
            fsms[ex] = BinaryFSM(s, m)
    return fsms


def generate_and_save_report(
        out_path: str, source: str, total_frames: int, no_person_frames: int,
        detected_history: List[str], fsms: Dict[str, BinaryFSM], all_feedback: set, is_live: bool
) -> None:
    """Generates the final JSON rating report based on the workout session."""
    status = "ok"
    final_ex = None
    reps = 0
    rating = None

    if total_frames == 0 or (no_person_frames / total_frames > 0.9):
        status = "no_person_detected"
    elif len(detected_history) < MIN_FRAMES_FOR_VALID_EXERCISE:
        status = "unknown_exercise"
    else:
        final_ex = Counter(detected_history).most_common(1)[0][0]
        if final_ex in fsms:
            reps = fsms[final_ex].counter

        if reps > 0:
            rating = max(1, 5 - len(all_feedback))

    report = {
        "video_path": "LIVE SESSION" if is_live else source,
        "status": status,
        "exercise": final_ex,
        "reps": reps,
        "feedback": list(all_feedback)
    }

    if rating is not None:
        report["rating"] = rating

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n[SUCCESS] Final report saved to: {out_path}")


def process_video(source: Optional[str], out_path: str, is_live: bool = False, camera_idx: int = 0,
                  show_window: bool = True) -> None:
    """Main execution loop for analyzing workout video feeds."""
    print(f"[INFO] Starting analysis. Mode: {'LIVE' if is_live else 'FILE'}")

    base_dir = Path(__file__).resolve().parent.parent.parent
    paths = get_system_paths(base_dir)

    try:
        # Load all models and configurations
        count_model = load_model(paths["count_model"])
        count_scaler = joblib.load(paths["count_scaler"])
        count_enc = joblib.load(paths["count_enc"])

        phase_idx_map = {name: i for i, name in enumerate(count_enc.classes_)}

        hybrid_feeder_model = load_model(paths["hybrid_feeder_model"])
        hybrid_feeder_scaler = joblib.load(paths["hybrid_feeder_scaler"])

        full_old_model = load_model(paths["old_seq_model"])
        motion_extractor = tf.keras.models.Sequential(full_old_model.layers[:-2])
        old_scaler = joblib.load(paths["old_seq_scaler"])

        with open(paths["feature_cols"], "r", encoding="utf-8") as f:
            feature_cols = json.load(f)

        hybrid_model = load_model(paths["hybrid_model"])
        ex_enc = joblib.load(paths["ex_enc"])
        exercise_classes = list(ex_enc.classes_)

    except Exception as e:
        print(f"[ERROR] Critical model loading error: {e}")
        return

    fsms = init_fsm_machines(phase_idx_map)
    evaluator = ExerciseEvaluator()
    abstainer = SmoothAbstain(len(exercise_classes))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(camera_idx if is_live else source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    # Session State tracking
    total_frames, no_person_frames = 0, 0
    buf_pose, buf_motion = deque(maxlen=SEQ_LEN), deque(maxlen=SEQ_LEN)
    detected_history, all_feedback = [], set()

    current_ex_display = "Scanning..."
    current_reps_display = 0
    current_feedback_display = ""

    while True:
        ret, frame = cap.read()
        if not ret: break

        if is_live:
            frame = cv2.flip(frame, 1)

        total_frames += 1
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        person_detected = False

        if res.pose_landmarks:
            core_points = [res.pose_landmarks.landmark[i] for i in (11, 12, 23, 24)]
            avg_visibility = sum(p.visibility for p in core_points) / 4.0
            if avg_visibility > MIN_VISIBILITY_THRESHOLD:
                person_detected = True

        if not person_detected:
            no_person_frames += 1
            buf_pose.clear()
            buf_motion.clear()
            draw_no_person_warning(frame)
        else:
            # 1. Inference Pipeline
            x99_counting = preprocess_pose_99(res.pose_landmarks.landmark, count_scaler)
            probs_phases = count_model.predict(x99_counting, verbose=0)[0]

            x99_hybrid = preprocess_pose_99(res.pose_landmarks.landmark, hybrid_feeder_scaler)
            probs_classes = hybrid_feeder_model.predict(x99_hybrid, verbose=0)[0]

            # 2. Update Rep Counters
            for machine in fsms.values():
                machine.update(probs_phases, phase_idx_map)

            # 3. Buffer Data (applying temperature scaling to probabilities)
            probs_classes = probs_classes ** 2.0
            s = np.sum(probs_classes)
            if s > 0: probs_classes /= s
            buf_pose.append(probs_classes)

            x170 = extract_features_170(res, feature_cols)
            if x170 is not None:
                buf_motion.append(old_scaler.transform([x170])[0])

            # 4. Trigger Hybrid Model if buffers are full
            if len(buf_pose) == SEQ_LEN and len(buf_motion) == SEQ_LEN:
                in_p = np.expand_dims(np.array(buf_pose)[::STEP], 0)
                in_m = np.expand_dims(np.array(buf_motion)[::STEP], 0)

                emb = motion_extractor.predict(in_m, verbose=0)
                final_probs = hybrid_model.predict([in_p, emb], verbose=0)[0]
                idx = abstainer.update_and_decide(final_probs)

                if idx != "unknown":
                    current_ex_display = exercise_classes[idx]
                    detected_history.append(current_ex_display)
                    current_reps_display = fsms[current_ex_display].counter if current_ex_display in fsms else 0

                    errs = evaluator.evaluate(res.pose_landmarks.landmark, current_ex_display)
                    current_feedback_display = errs[0] if errs else "Good form"
                    all_feedback.update(errs)
                else:
                    current_ex_display = "Unknown"

            # 5. Draw Skeleton & HUD
            mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            draw_workout_hud(frame, current_ex_display.upper(), current_reps_display, current_feedback_display)

        if show_window:
            cv2.imshow('AI Workout Assistant', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Interrupted by user.")
                break
        else:
            if total_frames % 30 == 0:
                print(".", end="", flush=True)

    cap.release()
    cv2.destroyAllWindows()

    generate_and_save_report(
        out_path, source, total_frames, no_person_frames,
        detected_history, fsms, all_feedback, is_live
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Workout Assistant Backend")
    ap.add_argument("--mode", choices=["file", "live"], default="file", help="Execution mode")
    ap.add_argument("--path", help="Path to video file (required if mode is file)")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--camera_index", type=int, default=0, help="Camera index for live mode")

    args = ap.parse_args()

    if args.mode == "file" and not args.path:
        print("[ERROR] In 'file' mode, you must provide --path.")
        exit(1)

    is_live_mode = (args.mode == "live")

    process_video(
        source=args.path if not is_live_mode else None,
        out_path=args.out,
        is_live=is_live_mode,
        camera_idx=args.camera_index,
        show_window=is_live_mode
    )