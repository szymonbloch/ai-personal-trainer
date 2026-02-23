import pandas as pd
from pathlib import Path
from typing import Optional

# Define class ranges as a constant at the module level
CLASS_RANGES = [
    (0, 106, "jumping_jack"),
    (107, 207, "pull_up"),
    (208, 306, "push_up"),
    (307, 384, "situp"),
    (385, 447, "squat")
]

def get_class(vid_id: int) -> Optional[str]:
    """
    Determines the exercise class based on the video ID.

    Args:
        vid_id (int): The ID of the video.

    Returns:
        Optional[str]: The name of the exercise class, or None if not found.
    """
    for start, end, cls in CLASS_RANGES:
        if start <= vid_id <= end:
            return cls
    return None

def load_and_merge_sequence_data(dataset_dir: Path) -> pd.DataFrame:
    """
    Loads sequence CSV files, preprocesses them, and merges into a single DataFrame.

    Args:
        dataset_dir (Path): The directory path containing the dataset files.

    Returns:
        pd.DataFrame: The merged sequence DataFrame.

    Raises:
        FileNotFoundError: If any of the required CSV files are missing.
    """
    files_to_load = {
        "landmarks": dataset_dir / "landmarks.csv",
        "angles": dataset_dir / "angles.csv",
        "distances": dataset_dir / "calculated_3d_distances.csv",
        "xyz_distances": dataset_dir / "xyz_distances.csv"
    }

    # Verify all required files exist before attempting to read them
    for name, path in files_to_load.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    landmarks = pd.read_csv(files_to_load["landmarks"])
    angles = pd.read_csv(files_to_load["angles"])
    distances = pd.read_csv(files_to_load["distances"])
    xyz_distances = pd.read_csv(files_to_load["xyz_distances"])

    dataframes = [landmarks, angles, distances, xyz_distances]

    # Ensure consistency by explicitly casting key columns to integer type
    for df in dataframes:
        df['vid_id'] = df['vid_id'].astype(int)
        df['frame_order'] = df['frame_order'].astype(int)

    # Merge all dataframes sequentially on vid_id and frame_order
    merged_df = (
        landmarks
        .merge(angles, on=['vid_id', 'frame_order'], how='inner')
        .merge(distances, on=['vid_id', 'frame_order'], how='inner')
        .merge(xyz_distances, on=['vid_id', 'frame_order'], how='inner')
    )

    return merged_df

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns class labels to the DataFrame based on video IDs.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'vid_id' column.

    Returns:
        pd.DataFrame: The DataFrame with the newly assigned 'class' column.
    """
    df['class'] = df['vid_id'].apply(get_class)
    return df

def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (Path): The target file path for the CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Successfully saved merged sequence data to: {output_path}")

def main() -> None:
    """Main execution function for the script."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "datasets" / "sequence_exercises_dataset"
    output_path = dataset_dir / "merged_sequence_data.csv"

    try:
        merged_data = load_and_merge_sequence_data(dataset_dir)
        labeled_data = assign_labels(merged_data)
        save_dataset(labeled_data, output_path)
    except Exception as e:
        print(f"An error occurred while processing sequence datasets: {e}")

if __name__ == "__main__":
    main()
