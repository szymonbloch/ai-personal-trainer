from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import joblib, json
from pathlib import Path
import random
import os

# Ustawienie ziarna losowości dla powtarzalności wyników
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = np.load("../../datasets/ready_to_train/sequence_dataset_preprocessed.npz")
X_train_raw, y_train = data["X_train"], data["y_train"]
X_val_raw,   y_val   = data["X_val"],   data["y_val"]
X_test_raw,  y_test  = data["X_test"],  data["y_test"]

# ---------------------------------------------------------
# KLUCZOWY HACK: TIME DILATION (Rozciąganie czasu)
# Bierzemy co 2. klatkę (indeksy 0, 2, 4, 6, 8 ...)
# Zamiast 30 klatek (1 sekunda), mamy 15 klatek reprezentujących 2 sekundy.
# To pozwoli modelowi zobaczyć cały przysiad (dół i góra).
# ---------------------------------------------------------

step = 2
X_train = X_train_raw[:, ::step, :]
X_val   = X_val_raw[:, ::step, :]
X_test  = X_test_raw[:, ::step, :]

print(f"Nowy kształt danych: {X_train.shape}")

encoder_a = joblib.load("../../model/sequence_model_LSTM/sequence_encoder.pkl")

with open("../../datasets/ready_to_train/sequence_feature_cols.json") as f:
    feature_cols = json.load(f)

n_classes = len(encoder_a.classes_)
T = X_train.shape[1]  # długość sekwencji (30)
F = X_train.shape[2]  # liczba cech (170)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Tworzymy słownik wag {0: 1.0, 1: 0.8, ...}
class_weights_dict = dict(enumerate(class_weights))

print("Wyliczone wagi dla klas:", class_weights_dict)
# Klasa z najmniejszą liczbą próbek (squat) powinna dostać najwyższą wagę (>1.0)
# ZMNIEJSZAMY sieć o połowę (z 64 na 32 filtry/neurony).
# Mniejszy model trudniej przeuczyć.

model_a = Sequential([
    # 1. PRZYWRACAMY SZUM. Bez tego model wkuwa na pamięć.
    # 0.025 to wartość, która zmusi go do szukania ogólnych kształtów.
    GaussianNoise(0.025, input_shape=(T, F)),
    BatchNormalization(),

    # 2. ZMNIEJSZAMY LSTM.
    # 64 neurony przy 15 klatkach to za dużo -> przeuczenie.
    # Wracamy do 32. Mniej neuronów = mniej miejsca na zapamiętywanie śmieci.
    Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4, return_sequences=False)),

    # 3. Zwiększamy Dropout i L2
    Dense(32, activation='relu', kernel_regularizer=l2(0.02)),  # Mocne L2
    Dropout(0.6),  # Bardzo agresywny Dropout (wyłączamy ponad połowę neuronów)

    Dense(n_classes, activation='softmax')
])

# Wolniejszy learning rate, żeby nie wystrzelić w kosmos z Loss
optimizer = Adam(learning_rate=0.0001)
model_a.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Dodajemy ReduceLROnPlateau
callbacks = [
    # Jeśli val_loss nie spadnie przez 3 epoki, zmniejsz LR o połowę (factor=0.5)
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
    # EarlyStopping czuwa, żeby nie trenować w nieskończoność
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('../../model/sequence_LSTM_model/sequence_model_final.keras', save_best_only=True)
]

# Trenujemy!
history = model_a.fit(
    X_train, y_train,
    epochs=50,  # Daj mu więcej epok, bo LR będzie maleć
    batch_size=32,
    validation_data=(X_val, y_val),
    # class_weight=class_weights_dict,  Wyłączone bo model przedobrzył (załatwione zostało to tym, że bierzemy co 2 klatkę)
    callbacks=callbacks
)

test_loss, test_acc = model_a.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

model_path = Path("../../model/sequence_LSTM_model/sequence-model2.keras")
model_a.save(model_path)

print("feature_cols:", feature_cols)
print("len(feature_cols):", len(feature_cols))
print("Saved model to:", model_path)

# 1. Pobranie predykcji
# y_pred_probs: prawdopodobieństwa (do ROC)
# y_pred_classes: konkretne klasy (do Macierzy Pomyłek)
y_pred_probs = model_a.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Lista nazw klas (jeśli encoder ma atrybut classes_, w przeciwnym razie numery)
class_names = encoder_a.classes_ if hasattr(encoder_a, 'classes_') else [str(i) for i in range(n_classes)]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# --- WYKRES 1: Krzywe uczenia (Loss & Accuracy) ---
def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Funkcja straty (Loss)')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Funkcja straty')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Dokładność')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
plot_learning_curves(history)

# --- WYKRES 2: Macierz Pomyłek (Znormalizowana) ---
def plot_confusion_matrix(y_true, y_pred, classes):
    # Oblicz macierz
    cm = confusion_matrix(y_true, y_pred)
    # Normalizacja (wiersze sumują się do 1)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Znormalizowana Macierz Pomyłek')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.show()
plot_confusion_matrix(y_test, y_pred_classes, class_names)

# --- WYKRES 3: Raport Klasyfikacji (Heatmapa) ---
def plot_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose().iloc[:-3, :3]  # Usunięcie średnich, zostawienie klas

    plt.figure(figsize=(8, len(classes) * 0.5 + 2))
    sns.heatmap(df_report, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Raport Klasyfikacji')
    plt.show()
plot_classification_report(y_test, y_pred_classes, class_names)

# --- WYKRES 4: Multiclass ROC Curve (Opcjonalnie) ---
# Działa najlepiej, gdy liczba klas nie jest ogromna (np. < 10-15)
def plot_multiclass_roc(y_test, y_pred_probs, classes):
    # Binaryzacja etykiet testowych (one-hot dla y_test)
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Klasa {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
    plt.legend(loc="lower right")
    plt.show()

# Uruchom tylko jeśli liczba klas jest rozsądna
if len(class_names) <= 10:
    plot_multiclass_roc(y_test, y_pred_probs, class_names)