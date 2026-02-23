import pandas as pd
from pathlib import Path


def analyze_pose_dataset(df: pd.DataFrame) -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the merged pose dataset.
    Prints basic information, missing values, statistics, and class distributions.

    Args:
        df (pd.DataFrame): The merged pose dataset to analyze.
    """
    print("=== Basic DataFrame Information ===")
    print(df.info())

    print("\n=== Missing Values Check ===")
    print(df.isnull().sum())

    print("\n=== Basic Numerical Statistics ===")
    print(df.describe())

    print("\n=== Class Distribution ===")
    print(df['pose'].value_counts())

    print("\n=== Number of Frames per Pose ID ===")
    print(df.groupby('pose_id').size())

    print("\n=== Nose Coordinates Statistics ===")
    nose_cols = ['x_nose', 'y_nose', 'z_nose']
    if all(col in df.columns for col in nose_cols):
        print(df[nose_cols].describe())
    else:
        print("Warning: Nose coordinate columns not found.")

    # Obliczanie liczby punktów (odrzucamy kolumny identyfikacyjne, zostają same współrzędne)
    landmark_cols = [c for c in df.columns if c not in ["pose_id", "pose"]]
    n_landmarks = len(landmark_cols) // 3
    print(f"\nCalculated number of 3D landmarks: {n_landmarks} (Expected: 33)")

    print("\n=== First and Last Rows for Each Class ===")
    for cls in df['pose'].unique():
        subset = df[df['pose'] == cls]
        print(f"\n--- Class: {cls} ---")
        print("First 3 rows:")
        print(subset.head(3))
        print("Last 3 rows:")
        print(subset.tail(3))

    print("\n=== Duplicates Check ===")
    duplicates = df.duplicated(subset=['pose_id', 'pose'])
    print(f"Number of duplicated rows: {duplicates.sum()}")


def main() -> None:
    """Main execution function for dataset validation."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "datasets" / "pose_exercises_dataset" / "merged_pose_data.csv"

    try:
        print(f"Loading dataset from: {csv_path.name}...\n")
        merged_df = pd.read_csv(csv_path)
        analyze_pose_dataset(merged_df)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()