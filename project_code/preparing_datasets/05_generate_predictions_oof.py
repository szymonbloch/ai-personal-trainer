import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    BatchNormalization, GaussianNoise, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Global configuration constants
SEQ_LEN = 30
STEP = 2
N_SPLITS = 5
RANDOM_STATE = 42


def create_pose_model(input_dim: int, n_classes: int) -> Sequential:
    """Builds and compiles the base MLP model for pose features."""
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


def create_motion_model(input_shape: Tuple[int, int], n_classes: int) -> Model:
    """
    Builds and compiles the Bidirectional LSTM model for motion features,
    including a designated bottleneck feature layer.

    """
    inputs = Input(shape=input_shape)

    x = GaussianNoise(0.025)(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.0, return_sequences=False))(x)

    # Bottleneck layer (Meta-features extraction point)
    x = Dense(32, kernel_regularizer=l2(0.02), name="feature_layer")(x)
    x = Activation('relu')(x)
    x = Dropout(0.6)(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_temporal_windows(
        x_data: np.ndarray, y_data: np.ndarray, vid_ids: np.ndarray, seq_len: int, step_dilation: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms continuous frame data into overlapping temporal windows per video.
    """
    x_out, y_out = [], []
    unique_vids = np.unique(vid_ids)

    for vid in unique_vids:
        indices = np.where(vid_ids == vid)[0]
        data_vid = x_data[indices]
        y_vid = y_data[indices]

        if len(data_vid) >= seq_len:
            for i in range(0, len(data_vid) - seq_len + 1):
                win = data_vid[i: i + seq_len]
                if step_dilation > 1:
                    win = win[::step_dilation]
                x_out.append(win)
                y_out.append(y_vid[i + seq_len - 1])

    return np.array(x_out), np.array(y_out)


def extract_hybrid_oof_features(
        pose_probs: np.ndarray,
        motion_feats: np.ndarray,
        y_true: np.ndarray,
        groups: np.ndarray,
        feat_extractor: Model
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[Any]]:
    """
    Generates Out-Of-Fold hybrid features (Pose probabilities + Motion embeddings)
    using sliding windows over validation data.
    """
    x_seq_list, x_motion_list, y_list, groups_list = [], [], [], []
    unique_vids = np.unique(groups)

    for vid in unique_vids:
        indices = np.where(groups == vid)[0]
        p_probs_vid = pose_probs[indices]
        m_feats_vid = motion_feats[indices]
        y_vid_frames = y_true[indices]

        if len(p_probs_vid) >= SEQ_LEN:
            vid_seq_probs, vid_mot_windows, vid_y = [], [], []

            for i in range(0, len(p_probs_vid) - SEQ_LEN + 1):
                vid_seq_probs.append(p_probs_vid[i: i + SEQ_LEN])
                vid_mot_windows.append(m_feats_vid[i: i + SEQ_LEN][::STEP])
                vid_y.append(y_vid_frames[i + SEQ_LEN - 1])

            if vid_seq_probs:
                vid_mot_windows = np.array(vid_mot_windows)
                mot_embeddings = feat_extractor.predict(vid_mot_windows, verbose=0)

                x_seq_list.extend(vid_seq_probs)
                x_motion_list.extend(mot_embeddings)
                y_list.extend(vid_y)
                groups_list.extend([vid] * len(vid_y))

    return x_seq_list, x_motion_list, y_list, groups_list


def visualize_oof_results(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_dir: Path) -> None:
    """Plots and saves diagnostic charts for the Out-Of-Fold procedure."""
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Confusion Matrix (Counts)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('OOF Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "oof_confusion_matrix_counts.png")
    plt.close()

    # 2. Confusion Matrix (Normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('OOF Normalized Confusion Matrix (Recall)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / "oof_confusion_matrix_norm.png")
    plt.close()

    # 3. Classification Report Heatmap
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().iloc[:-3, :3]

    plt.figure(figsize=(8, len(class_names) * 0.8 + 2))
    sns.heatmap(df_report, annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=1)
    plt.title('OOF Classification Report (Precision / Recall / F1)')
    plt.tight_layout()
    plt.savefig(save_dir / "oof_classification_report.png")
    plt.close()


def main() -> None:
    """Main execution pipeline for generating OOF datasets."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    in_raw = base_dir / "datasets" / "ready_to_train" / "raw_features_all.npz"
    in_encoder = base_dir / "datasets" / "ready_to_train" / "label_encoder_oof.pkl"
    out_oof = base_dir / "datasets" / "ready_to_train" / "oof_dataset.npz"
    out_plots_dir = base_dir / "plots" / "oof_diagnostics"

    print(f"Loading data from {in_raw.name}...")
    data = np.load(in_raw, allow_pickle=True)
    x_pose, x_motion, y, groups = data["X_pose"], data["X_motion"], data["y"], data["groups"]

    try:
        encoder = joblib.load(in_encoder)
        class_names = list(encoder.classes_)
        print(f"Loaded class names: {class_names}")
    except FileNotFoundError:
        print("WARNING: Encoder not found, using numeric class labels.")
        class_names = [str(i) for i in np.unique(y)]

    classes_unique = np.unique(y)
    n_classes = len(classes_unique)
    print(f"Classes: {n_classes} | Unique Videos: {len(np.unique(groups))}")

    # Containers for final OOF dataset
    final_x_seq, final_x_motion, final_y, final_groups = [], [], [], []

    # Containers for Diagnostics
    diag_true_labels, diag_pred_labels = [], []

    #
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(x_pose, y, groups), 1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        x_pose_tr, y_tr = x_pose[train_idx], y[train_idx]
        x_pose_val, y_val = x_pose[val_idx], y[val_idx]
        x_motion_tr, x_motion_val = x_motion[train_idx], x_motion[val_idx]
        groups_tr, groups_val = groups[train_idx], groups[val_idx]

        class_weights = dict(enumerate(compute_class_weight('balanced', classes=classes_unique, y=y_tr)))

        scaler_pose = StandardScaler()
        x_pose_tr = scaler_pose.fit_transform(x_pose_tr)
        x_pose_val = scaler_pose.transform(x_pose_val)

        scaler_mot = StandardScaler()
        x_motion_tr = scaler_mot.fit_transform(x_motion_tr)
        x_motion_val = scaler_mot.transform(x_motion_val)

        # 1. Train Pose Model
        print("  -> Training Pose Model...")
        model_pose = create_pose_model(x_pose_tr.shape[1], n_classes)
        es_pose = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_pose.fit(
            x_pose_tr, y_tr, validation_data=(x_pose_val, y_val),
            epochs=30, batch_size=64, class_weight=class_weights,
            callbacks=[es_pose], verbose=0
        )

        # 2. Train Motion Model
        print("  -> Training Motion Model...")
        x_mot_seq_tr, y_mot_seq_tr = create_temporal_windows(x_motion_tr, y_tr, groups_tr, SEQ_LEN, STEP)
        x_mot_seq_val, y_mot_seq_val = create_temporal_windows(x_motion_val, y_val, groups_val, SEQ_LEN, STEP)

        model_mot = create_motion_model((x_mot_seq_tr.shape[1], x_mot_seq_tr.shape[2]), n_classes)
        es_mot = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_mot.fit(
            x_mot_seq_tr, y_mot_seq_tr, validation_data=(x_mot_seq_val, y_mot_seq_val),
            epochs=30, batch_size=64, class_weight=class_weights,
            callbacks=[es_mot], verbose=0
        )

        feat_extractor = Model(inputs=model_mot.input, outputs=model_mot.get_layer("feature_layer").output)

        # 3. Generate OOF Predictions
        print("  -> Generating OOF predictions...")
        pose_probs_val = model_pose.predict(x_pose_val, verbose=0)

        diag_true_labels.extend(y_val)
        diag_pred_labels.extend(np.argmax(pose_probs_val, axis=1))

        seq_list, mot_list, y_list, grp_list = extract_hybrid_oof_features(
            pose_probs_val, x_motion_val, y_val, groups_val, feat_extractor
        )

        final_x_seq.extend(seq_list)
        final_x_motion.extend(mot_list)
        final_y.extend(y_list)
        final_groups.extend(grp_list)

    print("\n" + "=" * 50)
    print(" OOF DIAGNOSTIC REPORT ")
    print("=" * 50)
    print(classification_report(diag_true_labels, diag_pred_labels, target_names=class_names))

    print(f"\nGenerating plots in: {out_plots_dir.name}/")
    visualize_oof_results(diag_true_labels, diag_pred_labels, class_names, out_plots_dir)

    print(f"Saving OOF dataset to: {out_oof.name}...")
    np.savez_compressed(
        out_oof,
        X_seq=np.array(final_x_seq, dtype=np.float32),
        X_motion=np.array(final_x_motion, dtype=np.float32),
        y_seq=np.array(final_y),
        groups=np.array(final_groups)
    )
    print("SUCCESS: OOF Dataset ready and plots generated!")


if __name__ == "__main__":
    main()