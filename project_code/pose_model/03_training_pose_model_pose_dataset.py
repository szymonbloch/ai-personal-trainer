import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Global configurations
RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def load_dataset_split(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a CSV dataset split and separates features from labels.

    Args:
        csv_path (Path): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y).
    """
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y


def build_model(n_features: int, n_classes: int) -> Sequential:
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) model for pose classification.

    Args:
        n_features (int): Number of input features.
        n_classes (int): Number of target classes.

    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
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


def plot_learning_curves(history: tf.keras.callbacks.History, save_path: Path) -> None:
    """Plots and saves training and validation loss and accuracy curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix_custom(
        y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: Path
) -> None:
    """Plots and saves a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_classification_report_heatmap(
        y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], save_path: Path
) -> None:
    """Plots and saves a heatmap of precision, recall, and F1-score."""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df_classes = df.drop(['accuracy', 'macro avg', 'weighted avg'])[['precision', 'recall', 'f1-score']]
    df_classes.columns = ['Precision', 'Recall', 'F1-Score']

    plt.figure(figsize=(10, len(classes) * 0.6 + 2))
    sns.heatmap(df_classes, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    plt.title('Classification Report')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multiclass_roc(
        y_true: np.ndarray, y_probs: np.ndarray, n_classes: int, classes: List[str], save_path: Path
) -> None:
    """Plots and saves the Receiver Operating Characteristic (ROC) curve for multi-class."""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Main execution pipeline for training and evaluating the MLP pose model."""
    set_seeds()
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})

    # Directory configuration
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "datasets" / "ready_to_train" / "pose_model_MLP"
    model_dir = base_dir / "model" / "pose_model_MLP"
    plots_dir = base_dir / "output" / "plots" / "pose_model_MLP"

    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "pose_model_pose_dataset.keras"
    encoder_path = model_dir / "pose_model_encoder_pose_dataset.pkl"

    try:
        # 1. Load Data
        print("Loading datasets...")
        X_train, y_train = load_dataset_split(data_dir / "pose_train.csv")
        X_val, y_val = load_dataset_split(data_dir / "pose_val.csv")
        X_test, y_test = load_dataset_split(data_dir / "pose_test.csv")

        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
        print(f"Number of classes: {n_classes}")

        # 2. Build and Train Model

        model = build_model(n_features, n_classes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        model.save(model_path)
        print(f"\nModel saved to: {model_path}")

        # 3. Evaluation & Plotting
        try:
            encoder = joblib.load(encoder_path)
            class_names = list(encoder.classes_)
            print(f"Successfully loaded class names: {class_names}")
        except FileNotFoundError:
            print("WARNING: Encoder not found. Using numeric class labels for plots.")
            class_names = [str(i) for i in range(n_classes)]

        print("\nGenerating evaluation predictions...")
        y_pred_probs = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print("Generating and saving plots...")
        plot_learning_curves(history, plots_dir / "learning_curves.png")

        plot_confusion_matrix_custom(y_test, y_pred_classes, class_names, plots_dir / "confusion_matrix.png")

        plot_classification_report_heatmap(y_test, y_pred_classes, class_names, plots_dir / "classification_report.png")

        plot_multiclass_roc(y_test, y_pred_probs, n_classes, class_names, plots_dir / "roc_curve.png")

        print(f"All evaluation plots saved to: {plots_dir}")

    except Exception as e:
        print(f"An error occurred during model training: {e}")


if __name__ == "__main__":
    main()