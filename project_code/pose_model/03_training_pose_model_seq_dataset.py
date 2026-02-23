import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Any

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, History
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# Global configuration constant
RANDOM_STATE = 42


def build_mlp_model(input_dim: int, n_classes: int) -> Sequential:
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) model.

    Args:
        input_dim (int): Number of input features.
        n_classes (int): Number of target classes.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def split_and_scale_data(
        X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Splits data using GroupShuffleSplit to prevent data leakage from the same video,
    and scales features strictly based on the training set distribution.


    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Labels array.
        groups (np.ndarray): Array of video IDs for grouping.

    Returns:
        Tuple: (X_train_scaled, X_val_scaled, y_train, y_val, fitted_scaler)
    """
    # Split 90% Train, 10% Val keeping video frames isolated
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit scaler ONLY on training data to prevent leakage, then transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculates balanced class weights to handle imbalanced datasets.

    Args:
        y_train (np.ndarray): Training labels.

    Returns:
        Dict[int, float]: Dictionary mapping class indices to their computed weights.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return dict(enumerate(weights))


def plot_learning_curves(history: History, save_path: Path) -> None:
    """Plots and saves the training and validation loss and accuracy curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy', linestyle="--")
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss', linestyle="--")
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix_custom(
        y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: Path
) -> None:
    """Plots and saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Validation Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Main execution pipeline for training the MLP model on sequence-derived datasets."""
    sns.set_style("whitegrid")

    # 1. Setup paths dynamically
    base_dir = Path(__file__).resolve().parent.parent.parent

    in_raw_path = base_dir / "datasets" / "ready_to_train" / "raw_features_all.npz"
    in_encoder_path = base_dir / "model" / "combined_model" / "label_encoder_oof.pkl"

    out_model_dir = base_dir / "model" / "pose_model_MLP"
    out_plots_dir = base_dir / "output" / "plots" / "pose_model_MLP"

    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_model_dir / "pose_model_seq_dataset.keras"
    scaler_path = out_model_dir / "pose_model_scaler_seq_dataset.pkl"
    encoder_path = out_model_dir / "pose_model_encoder_seq_dataset.pkl"

    try:
        # 2. Load Data
        print(f"Loading sequence data from: {in_raw_path.name}...")
        data = np.load(in_raw_path, allow_pickle=True)
        X = data["X_pose"]
        y = data["y"]
        groups = data["groups"]

        print(f"Input shape: {X.shape}, Unique classes: {len(np.unique(y))}")

        # 3. Load Encoder
        try:
            encoder = joblib.load(in_encoder_path)
            class_names = list(encoder.classes_)
            print(f"Loaded class names: {class_names}")
        except FileNotFoundError:
            print(f"Error: Encoder file not found at {in_encoder_path}")
            return

        # 4. Split and Scale Data
        print("Splitting data (GroupShuffleSplit) and scaling features...")
        X_train, X_val, y_train, y_val, scaler = split_and_scale_data(X, y, groups)

        print(f"  Train set: {len(X_train)} frames")
        print(f"  Val set:   {len(X_val)} frames (unseen video isolation)")

        class_weights = calculate_class_weights(y_train)
        print(f"  Class weights applied: {class_weights}")

        # 5. Build and Train Model
        print("\nStarting model training...")
        # Dynamically determine input dimension instead of hardcoding 99
        input_dim = X_train.shape[1]
        n_classes = len(class_names)

        model = build_mlp_model(input_dim, n_classes)

        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=64,
            class_weight=class_weights,
            callbacks=[es],
            verbose=1
        )

        # 6. Save Artifacts
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)
        print(f"\n[SUCCESS] Model and artifacts saved to {out_model_dir.name}/")

        # 7. Evaluate and Plot
        print("\nGenerating evaluation plots...")
        y_pred_probs = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print("\n=== CLASSIFICATION REPORT (Unseen Video Validation) ===")
        print(classification_report(y_val, y_pred_classes, target_names=class_names))

        plot_learning_curves(history, out_plots_dir / "learning_curves_seq_dataset.png")
        plot_confusion_matrix_custom(y_val, y_pred_classes, class_names,
                                     out_plots_dir / "confusion_matrix_seq_dataset.png")

        print(f"Plots saved successfully to: {out_plots_dir}")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")


if __name__ == "__main__":
    main()