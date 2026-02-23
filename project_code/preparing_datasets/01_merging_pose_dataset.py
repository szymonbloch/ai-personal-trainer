import pandas as pd
from pathlib import Path


def load_and_merge_pose_data(dataset_dir: Path) -> pd.DataFrame:
    """
    Loads landmarks and labels CSV files and merges them into a single
    DataFrame based on the 'pose_id' column.

    Args:
        dataset_dir (Path): The directory path containing the dataset files.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        FileNotFoundError: If the required CSV files are missing.
    """
    landmarks_path = dataset_dir / "landmarks.csv"
    labels_path = dataset_dir / "labels.csv"

    if not landmarks_path.exists():
        raise FileNotFoundError(f"File not found: {landmarks_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"File not found: {labels_path}")

    landmarks = pd.read_csv(landmarks_path)
    labels = pd.read_csv(labels_path)

    # Merge data. Previously commented out files (angles, xyz, d3) were removed
    # following the Clean Code principle: "Do not leave dead code".
    merged_df = landmarks.merge(labels, on="pose_id")

    return merged_df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves the processed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (Path): The target file path for the CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Successfully saved merged data to: {output_path}")


def main() -> None:
    """Main execution function for the script."""
    # Dynamically build the path (assuming the script is in project_code/preparing_datasets/)
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "datasets" / "pose_exercises_dataset"
    output_path = dataset_dir / "merged_pose_data.csv"

    try:
        merged_data = load_and_merge_pose_data(dataset_dir)
        save_dataset(merged_data, output_path)
    except Exception as e:
        print(f"An error occurred while merging datasets: {e}")


if __name__ == "__main__":
    main()