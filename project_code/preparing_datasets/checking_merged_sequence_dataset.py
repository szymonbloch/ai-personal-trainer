import pandas as pd
from pathlib import Path


def analyze_sequence_dataset(df: pd.DataFrame) -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the merged sequence dataset.
    Prints basic information, missing values, statistics, and class distributions.

    Args:
        df (pd.DataFrame): The merged sequence dataset to analyze.
    """
    print("=== Basic DataFrame Information ===")
    print(df.info())

    print("\n=== Missing Values Check ===")
    print(df.isnull().sum())

    print("\n=== Basic Numerical Statistics ===")
    print(df.describe())

    print("\n=== Class Distribution ===")
    print(df['class'].value_counts())

    print("\n=== Number of Frames per Video ID ===")
    frames_per_vid = df.groupby('vid_id').size()
    print(frames_per_vid)
    print(f"Maximum frames: {frames_per_vid.max()}, Minimum frames: {frames_per_vid.min()}")

    print("\n=== Nose Coordinates Statistics ===")
    nose_cols = ['x_nose', 'y_nose', 'z_nose']
    if all(col in df.columns for col in nose_cols):
        print(df[nose_cols].describe())
    else:
        print("Warning: Nose coordinate columns not found.")

    # Bezpieczniejsze liczenie cech: odrzucamy kolumny identyfikacyjne
    excluded_cols = {"vid_id", "frame_order", "class"}
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    # Precyzyjne liczenie tylko punktów 3D (szukamy kolumn zaczynających się od 'x_')
    x_coords = [c for c in df.columns if str(c).startswith('x_')]

    print(f"\nTotal number of feature columns: {len(feature_cols)}")
    print(f"Calculated number of 3D landmarks (based on 'x_' coordinates): {len(x_coords)}")

    print("\n=== First and Last Rows for Each Class ===")
    for cls in df['class'].unique():
        subset = df[df['class'] == cls]
        print(f"\n--- Class: {cls} ---")
        print("First 3 rows:")
        print(subset.head(3))
        print("Last 3 rows:")
        print(subset.tail(3))

    print("\n=== Duplicates Check ===")
    duplicates = df.duplicated(subset=['vid_id', 'frame_order'])
    print(f"Number of duplicated rows (same video & frame): {duplicates.sum()}")


def main() -> None:
    """Main execution function for sequence dataset validation."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "datasets" / "sequence_exercises_dataset" / "merged_sequence_data.csv"

    try:
        print(f"Loading dataset from: {csv_path.name}...\n")
        merged_df = pd.read_csv(csv_path)
        analyze_sequence_dataset(merged_df)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()