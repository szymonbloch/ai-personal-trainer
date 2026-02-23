import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    Concatenate, BatchNormalization, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Global configuration & Hyperparameters
TIME_STEP = 2
DROPOUT_RATE = 0.55
NOISE_STD = 0.02
L2_REG = 0.005
LEARNING_RATE = 0.0003
RANDOM_STATE = 42


def load_and_preprocess_data(npz_path: Path, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads Out-Of-Fold data and applies time dilation to the sequences.

    Args:
        npz_path (Path): Path to the .npz dataset file.
        step (int): Time dilation step (e.g., 2 means taking every 2nd frame).

    Returns:
        Tuple: (X_sequence, X_motion, y_labels, groups)
    """
    data = np.load(npz_path)
    x_seq = data["X_seq"]  # Shape: (N, 30, Features)
    x_motion = data["X_motion"]  # Shape: (N, 32)
    y_seq = data["y_seq"]  # Shape: (N,)
    groups = data["groups"]  # Shape: (N,)

    # Apply Time Dilation
    x_seq_dilated = x_seq[:, ::step, :]
    return x_seq_dilated, x_motion, y_seq, groups


def split_hybrid_data(
        x_seq: np.ndarray, x_motion: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Performs a 3-way split (Train/Val/Test) ensuring no video group leakage.


    Returns:
        Tuple of (Train_data, Val_data, Test_data) where each is a tuple of (X_seq, X_motion, y).
    """
    # 1. Split into Train (70%) and Temp (30%)
    splitter_1 = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=RANDOM_STATE)
    train_idxs, temp_idxs = next(splitter_1.split(x_seq, y, groups))

    x_seq_tr, x_mot_tr, y_tr = x_seq[train_idxs], x_motion[train_idxs], y[train_idxs]
    groups_temp = groups[temp_idxs]

    # 2. Split Temp into Validation (15%) and Test (15%)
    splitter_2 = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=RANDOM_STATE)
    val_idxs, test_idxs = next(splitter_2.split(x_seq[temp_idxs], y[temp_idxs], groups_temp))

    x_seq_val, x_mot_val, y_val = x_seq[temp_idxs][val_idxs], x_motion[temp_idxs][val_idxs], y[temp_idxs][val_idxs]
    x_seq_ts, x_mot_ts, y_ts = x_seq[temp_idxs][test_idxs], x_motion[temp_idxs][test_idxs], y[temp_idxs][test_idxs]

    return (x_seq_tr, x_mot_tr, y_tr), (x_seq_val, x_mot_val, y_val), (x_seq_ts, x_mot_ts, y_ts)


def build_hybrid_model(seq_shape: Tuple[int, int], motion_shape: Tuple[int,], n_classes: int) -> Model:
    """
    Builds a two-branch hybrid neural network combining sequence data and static motion features.


    Args:
        seq_shape: Shape of the sequence input (timesteps, features).
        motion_shape: Shape of the motion input (features,).
        n_classes: Number of target classes.

    Returns:
        Model: Compiled Keras Model.
    """
    reg = l2(L2_REG)

    # --- Branch 1: Sequence Processing (LSTM) ---
    input_pose = Input(shape=seq_shape, name="input_pose")
    x1 = GaussianNoise(NOISE_STD)(input_pose)
    x1 = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=reg))(x1)
    x1 = Dropout(DROPOUT_RATE)(x1)

    # --- Branch 2: Motion Feature Processing (Dense) ---
    input_motion = Input(shape=motion_shape, name="input_motion")
    x2 = GaussianNoise(NOISE_STD)(input_motion)
    x2 = Dense(32, activation="relu", kernel_regularizer=reg)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(DROPOUT_RATE)(x2)

    # --- Fusion & Classification ---
    combined = Concatenate()([x1, x2])
    z = Dense(32, activation='relu', kernel_regularizer=reg)(combined)
    z = Dropout(DROPOUT_RATE)(z)
    output = Dense(n_classes, activation='softmax', name="output_class")(z)

    model = Model(inputs=[input_pose, input_motion], outputs=output, name="Hybrid_OOF_Model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_evaluation_metrics(
        history: tf.keras.callbacks.History,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
        save_dir: Path
) -> None:
    """Plots and saves learning curves, confusion matrix, and classification report."""
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss', linestyle="--")
    ax1.set_title('Loss Function')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', linestyle="--")
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "hybrid_learning_curves.png")
    plt.close()

    # 2. Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_dir / "hybrid_confusion_matrix.png")
    plt.close()

    # 3. Classification Report Heatmap
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose().iloc[:-3, :3]
    df_report.columns = ['Precision', 'Recall', 'F1-Score']

    plt.figure(figsize=(8, len(classes) * 0.5 + 2))
    sns.heatmap(df_report, annot=True, cmap='viridis', fmt='.2f', cbar=True)
    plt.title("Classification Report")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "hybrid_classification_report.png")
    plt.close()


def main() -> None:
    """Main execution pipeline for the hybrid sequence model."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Setup Paths
    npz_path = base_dir / "datasets" / "ready_to_train" / "oof_dataset.npz"
    ex_encoder_path = base_dir / "model" / "combined_model" / "label_encoder_oof.pkl"
    out_dir = base_dir / "model" / "combined_model"
    plots_dir = base_dir / "plots" / "combined_model_diagnostics"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "combined_sequence_model.keras"
    enc_out_final = out_dir / "sequence_encoder_from_probs.pkl"

    try:
        # 1. Load Data & Preprocess
        print(f"Loading and dilating OOF data from: {npz_path.name}...")
        x_seq, x_motion, y, groups = load_and_preprocess_data(npz_path, TIME_STEP)

        encoder = joblib.load(ex_encoder_path)
        class_names = list(encoder.classes_)
        joblib.dump(encoder, enc_out_final)

        print(f"Data shapes -> Seq: {x_seq.shape}, Motion: {x_motion.shape}")

        # 2. Split Data
        print("Performing GroupShuffleSplit (Train/Val/Test)...")
        train_data, val_data, test_data = split_hybrid_data(x_seq, x_motion, y, groups)
        x_seq_tr, x_mot_tr, y_tr = train_data
        x_seq_val, x_mot_val, y_val = val_data
        x_seq_ts, x_mot_ts, y_ts = test_data

        print(f"Samples -> Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_ts)}")

        # Calculate class weights
        cw_values = compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr)
        class_weights = dict(zip(np.unique(y_tr), cw_values))

        # 3. Build Model
        print("\nBuilding Hybrid Neural Network...")
        model = build_hybrid_model(
            seq_shape=(x_seq_tr.shape[1], x_seq_tr.shape[2]),
            motion_shape=(x_mot_tr.shape[1],),
            n_classes=len(class_names)
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(filepath=out_model, save_best_only=True, monitor='val_loss')
        ]

        # 4. Train Model
        print("\nStarting Training Phase...")
        history = model.fit(
            x=[x_seq_tr, x_mot_tr],
            y=y_tr,
            epochs=100,
            batch_size=32,
            validation_data=([x_seq_val, x_mot_val], y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # 5. Evaluate & Plot
        print("\nGenerating final evaluation metrics...")
        y_pred_probs = model.predict([x_seq_ts, x_mot_ts])
        y_pred = np.argmax(y_pred_probs, axis=1)

        loss, acc = model.evaluate([x_seq_ts, x_mot_ts], y_ts, verbose=0)
        print(f"\n[FINAL RESULT] Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

        plot_evaluation_metrics(history, y_ts, y_pred, class_names, plots_dir)
        print(f"Diagnostic plots saved to: {plots_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()