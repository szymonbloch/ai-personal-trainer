import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Global configuration constants
HIP_INDEX = 24
NOSE_INDEX = 0
FOOT_INDEX = 32
RANDOM_STATE = 42


def normalize_pose_features(x_raw: np.ndarray, n_joints: int) -> np.ndarray:
    """
    Normalizes spatial coordinates by centering them around the hip
    and scaling them based on body height.

    Args:
        x_raw (np.ndarray): Raw feature array of shape (n_samples, n_features).
        n_joints (int): Number of joints calculated from features.

    Returns:
        np.ndarray: Normalized and flattened feature array.
    """
    x_reshaped = x_raw.reshape(len(x_raw), n_joints, 3)

    # Center coordinates relative to the hip
    hip_coords = x_reshaped[:, HIP_INDEX, :]
    x_reshaped = x_reshaped - hip_coords[:, np.newaxis, :]

    # Normalize by body height (distance between nose and foot)
    body_heights = np.linalg.norm(
        x_reshaped[:, NOSE_INDEX, :] - x_reshaped[:, FOOT_INDEX, :], axis=1
    )
    # Prevent division by zero if landmarks are completely missing/corrupted
    body_heights = np.where(body_heights == 0, 1, body_heights)
    x_reshaped = x_reshaped / body_heights[:, np.newaxis, np.newaxis]

    return x_reshaped.reshape(len(x_reshaped), -1)


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Splits the dataset into training (70%), validation (15%), and testing (15%) sets.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Labels array.

    Returns:
        Tuple containing split arrays:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_and_scale(
        x_splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
        y_splits: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], StandardScaler, LabelEncoder]:
    """
    Encodes categorical labels and applies standard scaling to features.

    Args:
        x_splits: Tuple of (X_train, X_val, X_test).
        y_splits: Tuple of (y_train, y_val, y_test).

    Returns:
        Processed splits, the fitted scaler, and the fitted encoder.
    """
    x_train, x_val, x_test = x_splits
    y_train, y_val, y_test = y_splits

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return (x_train_scaled, x_val_scaled, x_test_scaled), (y_train_enc, y_val_enc, y_test_enc), scaler, encoder


def save_split_to_csv(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    """Saves a data split (features and labels) to a CSV file."""
    df_out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df_out["label"] = y
    df_out.to_csv(output_path, index=False)


def save_metadata(meta_dict: Dict[str, Any], output_path: Path) -> None:
    """Saves metadata configuration to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)


def main() -> None:
    """Main execution pipeline for pose data preprocessing."""
    # Dynamically build paths based on script location
    base_dir = Path(__file__).resolve().parent.parent.parent

    raw_csv_path = base_dir / "datasets" / "pose_exercises_dataset" / "merged_pose_data.csv"
    out_dir_data = base_dir / "datasets" / "ready_to_train" / "pose_model_MLP"
    out_dir_model = base_dir / "model" / "pose_model_MLP"

    out_dir_data.mkdir(parents=True, exist_ok=True)
    out_dir_model.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_csv = out_dir_data / "pose_train.csv"
    val_csv = out_dir_data / "pose_val.csv"
    test_csv = out_dir_data / "pose_test.csv"

    scaler_path = out_dir_model / "pose_model_scaler_pose_dataset.pkl"
    encoder_path = out_dir_model / "pose_model_encoder_pose_dataset.pkl"
    meta_path = out_dir_model / "pose_model_meta_pose_dataset.json"

    try:
        # 1. Load data
        pose_df = pd.read_csv(raw_csv_path)
        landmark_cols = [c for c in pose_df.columns if c not in ["pose_id", "pose"]]
        x_raw = pose_df[landmark_cols].values
        y_raw = pose_df["pose"].values

        n_joints = len(landmark_cols) // 3
        print(f"Before preprocessing: X shape {x_raw.shape}, y shape {y_raw.shape}")

        # 2. Normalize and extract features
        x_processed = normalize_pose_features(x_raw, n_joints)

        # 3. Split data
        x_splits = split_data(x_processed, y_raw)

        print("\nSplit shapes:")
        print(f"  train: {x_splits[0].shape}, {len(x_splits[3])}")
        print(f"  val:   {x_splits[1].shape}, {len(x_splits[4])}")
        print(f"  test:  {x_splits[2].shape}, {len(x_splits[5])}")

        # 4. Encode labels and scale features
        x_scaled, y_encoded, scaler, encoder = encode_and_scale(
            x_splits[:3], x_splits[3:]
        )

        print(f"\nClasses: {list(encoder.classes_)}")
        print("After preprocessing (scaled):")
        print(f"  train: {x_scaled[0].shape}, {y_encoded[0].shape}")
        print(f"  val:   {x_scaled[1].shape}, {y_encoded[1].shape}")
        print(f"  test:  {x_scaled[2].shape}, {y_encoded[2].shape}")

        # 5. Save processed CSVs
        save_split_to_csv(x_scaled[0], y_encoded[0], train_csv)
        save_split_to_csv(x_scaled[1], y_encoded[1], val_csv)
        save_split_to_csv(x_scaled[2], y_encoded[2], test_csv)

        print("\nSaved CSVs:")
        print(f"  {train_csv}")
        print(f"  {val_csv}")
        print(f"  {test_csv}")

        # 6. Save model artifacts (Scaler, Encoder, Meta)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)

        meta = {
            "n_joints": int(n_joints),
            "hip_index": int(HIP_INDEX),
            "nose_index": int(NOSE_INDEX),
            "foot_index": int(FOOT_INDEX),
            "landmark_cols": landmark_cols,
        }
        save_metadata(meta, meta_path)

        print("\nArtifacts saved:")
        print(f"  Scaler  -> {scaler_path}")
        print(f"  Encoder -> {encoder_path}")
        print(f"  Meta    -> {meta_path}")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")


if __name__ == "__main__":
    main()