import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Global configuration constants
SEQUENCE_LENGTH = 30
RANDOM_STATE = 42


def split_video_ids(vid_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits video IDs into training (64%), validation (16%), and testing (20%) sets.
    The split is performed on video IDs, not frames, to prevent data leakage.

    Args:
        vid_ids (np.ndarray): Array of unique video IDs.

    Returns:
        Tuple containing split video IDs: (train_vids, val_vids, test_vids).
    """
    train_val_vids, test_vids = train_test_split(
        vid_ids, test_size=0.2, random_state=RANDOM_STATE
    )
    train_vids, val_vids = train_test_split(
        train_val_vids, test_size=0.2, random_state=RANDOM_STATE
    )
    return train_vids, val_vids, test_vids


def fit_transformers(
        df: pd.DataFrame, train_vids: np.ndarray, feature_cols: List[str]
) -> Tuple[StandardScaler, LabelEncoder]:
    """
    Fits the scaler exclusively on training data to prevent data leakage,
    and fits the encoder on all available classes.

    Args:
        df (pd.DataFrame): The complete sequence DataFrame.
        train_vids (np.ndarray): Array of training video IDs.
        feature_cols (List[str]): List of feature column names.

    Returns:
        Tuple[StandardScaler, LabelEncoder]: Fitted scaler and encoder.
    """
    df_train = df[df["vid_id"].isin(train_vids)]
    x_train_all = df_train[feature_cols].values

    scaler = StandardScaler().fit(x_train_all)
    encoder = LabelEncoder().fit(df["class"].values)

    return scaler, encoder


def create_sequences(
        df: pd.DataFrame,
        vid_ids: np.ndarray,
        scaler: StandardScaler,
        encoder: LabelEncoder,
        feature_cols: List[str],
        seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates sliding window sequences and corresponding labels for given video IDs.


    Args:
        df (pd.DataFrame): The complete sequence DataFrame.
        vid_ids (np.ndarray): Video IDs to process.
        scaler (StandardScaler): Fitted scaler for features.
        encoder (LabelEncoder): Fitted encoder for labels.
        feature_cols (List[str]): List of feature column names.
        seq_length (int): The number of frames per sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features array (X) and labels array (y).
    """
    x_seq, y_seq = [], []

    for vid in vid_ids:
        # Extract and sort frames chronologically for the current video
        df_vid = df[df["vid_id"] == vid].sort_values("frame_order")

        # Transform data using pre-fitted scaler and encoder
        x_vid = scaler.transform(df_vid[feature_cols].values)
        y_vid = encoder.transform(df_vid["class"].values)

        # Apply sliding window to create sequences
        for i in range(len(df_vid) - seq_length + 1):
            x_seq.append(x_vid[i: i + seq_length])
            # The label for the sequence is taken from the last frame of that window
            y_seq.append(y_vid[i + seq_length - 1])

    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64)


def main() -> None:
    """Main execution pipeline for sequence data preprocessing."""
    # Dynamically build paths based on script location
    base_dir = Path(__file__).resolve().parent.parent.parent

    input_csv = base_dir / "datasets" / "sequence_exercises_dataset" / "merged_sequence_data.csv"

    # Output directories
    out_dir_data = base_dir / "datasets" / "ready_to_train"
    out_dir_model = base_dir / "model" / "sequence_model_LSTM"

    out_dir_data.mkdir(parents=True, exist_ok=True)
    out_dir_model.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    out_npz = out_dir_data / "sequence_dataset_preprocessed.npz"
    scaler_path = out_dir_model / "sequence_scaler.pkl"
    encoder_path = out_dir_model / "sequence_encoder.pkl"
    feature_cols_path = out_dir_model / "sequence_feature_cols.json"

    try:
        # 1. Load data
        df_sequence = pd.read_csv(input_csv)
        feature_cols = [c for c in df_sequence.columns if c not in ["vid_id", "frame_order", "class"]]
        vids = df_sequence["vid_id"].unique()

        # 2. Split video IDs (crucial to split by video, not by frames)
        train_vids, val_vids, test_vids = split_video_ids(vids)
        print(f"Video splits - Train: {len(train_vids)}, Val: {len(val_vids)}, Test: {len(test_vids)}")

        # 3. Fit Scaler and Encoder
        scaler, encoder = fit_transformers(df_sequence, train_vids, feature_cols)
        print(f"Classes: {list(encoder.classes_)}")

        # 4. Generate sequences
        print(f"\nGenerating sequences (Window Length: {SEQUENCE_LENGTH})...")
        x_train, y_train = create_sequences(df_sequence, train_vids, scaler, encoder, feature_cols, SEQUENCE_LENGTH)
        x_val, y_val = create_sequences(df_sequence, val_vids, scaler, encoder, feature_cols, SEQUENCE_LENGTH)
        x_test, y_test = create_sequences(df_sequence, test_vids, scaler, encoder, feature_cols, SEQUENCE_LENGTH)

        print("\nSequence shapes (samples, timesteps, features):")
        print(f"  Train: {x_train.shape}, {y_train.shape}")
        print(f"  Val:   {x_val.shape}, {y_val.shape}")
        print(f"  Test:  {x_test.shape}, {y_test.shape}")

        # 5. Save all artifacts
        np.savez_compressed(
            out_npz,
            X_train=x_train, y_train=y_train,
            X_val=x_val, y_val=y_val,
            X_test=x_test, y_test=y_test
        )

        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)

        with open(feature_cols_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        print("\nArtifacts successfully saved:")
        print(f"  Dataset -> {out_npz}")
        print(f"  Scaler  -> {scaler_path}")
        print(f"  Encoder -> {encoder_path}")
        print(f"  Columns -> {feature_cols_path}")

    except Exception as e:
        print(f"An error occurred during sequence pipeline execution: {e}")


if __name__ == "__main__":
    main()