import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

# Global configuration constants
HIP_INDEX = 24
NOSE_INDEX = 0
FOOT_INDEX = 32


def normalize_geometric_features(x_chunk: np.ndarray) -> np.ndarray:
    """
    Applies geometric normalization to 3D pose landmarks (99 features).
    Centers coordinates around the hip and scales by body height.

    Args:
        x_chunk (np.ndarray): Raw pose features array.

    Returns:
        np.ndarray: Normalized and flattened pose features array.
    """
    n_samples = x_chunk.shape[0]
    # Reshape to (samples, 33 landmarks, 3 coordinates)
    x_3d = x_chunk.reshape(n_samples, 33, 3)

    # Center coordinates relative to the hip
    hip_coords = x_3d[:, HIP_INDEX, :]
    x_3d = x_3d - hip_coords[:, np.newaxis, :]

    # Normalize by body height (distance between nose and foot)
    body_heights = np.linalg.norm(
        x_3d[:, NOSE_INDEX, :] - x_3d[:, FOOT_INDEX, :], axis=1
    )
    # Prevent division by zero
    body_heights = np.where(body_heights == 0, 1.0, body_heights)
    x_3d = x_3d / body_heights[:, np.newaxis, np.newaxis]

    return x_3d.reshape(n_samples, -1)


def identify_pose_columns(pose_csv_path: Path, df_full: pd.DataFrame) -> List[str]:
    """
    Identifies raw pose columns by comparing the sequence dataframe
    with the headers of the original pose dataset.

    Args:
        pose_csv_path (Path): Path to the original pose CSV file.
        df_full (pd.DataFrame): The full sequence DataFrame.

    Returns:
        List[str]: A list of valid pose column names.
    """
    # Read only headers to save memory
    df_headers = pd.read_csv(pose_csv_path, nrows=0)
    excluded_cols = {"pose", "pose_id", "label", "class"}

    pose_candidates = [c for c in df_headers.columns if c not in excluded_cols]
    pose_cols = [c for c in pose_candidates if c in df_full.columns]

    return pose_cols


def load_motion_columns(json_path: Path) -> List[str]:
    """Loads engineered motion feature column names from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_and_encode_data(
        df_full: pd.DataFrame, pose_cols: List[str], motion_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Extracts feature arrays from the DataFrame and encodes target labels.


[Image of Feature Extraction process in Machine Learning]


    Args:
        df_full (pd.DataFrame): The complete sorted DataFrame.
        pose_cols (List[str]): List of raw pose column names.
        motion_cols (List[str]): List of engineered motion column names.

    Returns:
        Tuple: (X_pose_raw, X_motion_raw, y_encoded, groups, fitted_encoder)
    """
    try:
        x_pose_raw = df_full[pose_cols].values.astype(np.float32)
    except KeyError as e:
        raise KeyError(f"Missing pose columns in the dataset: {e}")

    try:
        x_motion_raw = df_full[motion_cols].values.astype(np.float32)
    except KeyError as e:
        raise KeyError(f"Missing motion columns in the dataset: {e}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df_full["class"].values)
    groups = df_full["vid_id"].values

    return x_pose_raw, x_motion_raw, y_encoded, groups, encoder


def main() -> None:
    """Main execution pipeline for generating OOF-ready features."""
    # Dynamically build paths
    base_dir = Path(__file__).resolve().parent.parent.parent

    csv_sequence = base_dir / "datasets" / "sequence_exercises_dataset" / "merged_sequence_data.csv"
    csv_pose = base_dir / "datasets" / "pose_exercises_dataset" / "merged_pose_data.csv"
    feature_cols_json = base_dir / "datasets" / "ready_to_train" / "sequence_feature_cols.json"

    out_dir = base_dir / "datasets" / "ready_to_train"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_raw_npz = out_dir / "raw_features_all.npz"
    enc_out_pkl = out_dir / "label_encoder_oof.pkl"

    try:
        print(f"Loading main dataset from: {csv_sequence.name}...")
        df_full = pd.read_csv(csv_sequence)

        # Sort chronologically to preserve sequence integrity
        sort_keys = ["vid_id", "frame_order"] if "frame_order" in df_full.columns else ["vid_id"]
        df_full = df_full.sort_values(sort_keys)
        print(f"Loaded {len(df_full)} rows.")

        # 1 & 2. Identify Columns
        pose_cols = identify_pose_columns(csv_pose, df_full)
        print(f"Identified {len(pose_cols)} raw pose columns.")

        motion_cols = load_motion_columns(feature_cols_json)
        print(f"Identified {len(motion_cols)} engineered motion columns.")

        # 3. Extract Arrays & Encode
        print("Extracting arrays and encoding labels...")
        x_pose_raw, x_motion_raw, y, groups, encoder = extract_and_encode_data(
            df_full, pose_cols, motion_cols
        )

        # 4. Geometric Preprocessing
        print("Applying geometric normalization to pose features...")
        x_pose_processed = normalize_geometric_features(x_pose_raw)

        # 5. Save Artifacts
        print(f"Saving compiled features to: {out_raw_npz.name}...")
        np.savez_compressed(
            out_raw_npz,
            X_pose=x_pose_processed,
            X_motion=x_motion_raw,
            y=y,
            groups=groups
        )

        joblib.dump(encoder, enc_out_pkl)
        print("SUCCESS: Data is ready for Out-Of-Fold (OOF) procedures.")

    except Exception as e:
        print(f"An error occurred during dataset preparation: {e}")


if __name__ == "__main__":
    main()