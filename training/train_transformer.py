"""
train_transformer.py — Temporal Transformer загвараар LSTM-тэй харьцуулалт.

Ажиллуулах:
    python -m training.train_transformer --hand left
    python -m training.train_transformer --hand right
    python -m training.train_transformer --hand both   # хоёуланг нэгэн зэрэг

Гаралт:
    models/left_transformer.keras
    models/right_transformer.keras
    metrics/transformer_comparison.txt
    metrics/transformer_comparison.png
"""

import argparse
import os
import sys
import time
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
METRIC_DIR = os.path.join(BASE_DIR, "metrics")
SEQ_LEN    = 40
FEATURE_SIZE = 63

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Transformer блок
# ---------------------------------------------------------------------------

def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int,
                               dropout: float = 0.1):
    """Нэг Transformer encoder блок: Multi-Head Attention + FFN + residual."""
    # Multi-Head Self-Attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                               dropout=dropout)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-Forward Network
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(d_model)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(num_classes: int, d_model: int = 64,
                       num_heads: int = 4, ff_dim: int = 128,
                       num_blocks: int = 2, dropout: float = 0.1) -> Model:
    """
    Temporal Transformer for gesture sequence classification.
    Input: (batch, SEQ_LEN, FEATURE_SIZE)
    Output: (batch, num_classes)
    """
    inp = Input(shape=(SEQ_LEN, FEATURE_SIZE))

    # Feature projection
    x = Dense(d_model)(inp)
    x = Dropout(dropout)(x)

    # Transformer encoder стэк
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout)

    # Sequence aggregation
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str, classes: list):
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".npy"):
                continue
            seq = np.load(os.path.join(cls_dir, fname))
            if seq.shape != (SEQ_LEN, FEATURE_SIZE):
                continue
            X.append(seq)
            y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------------
# Нэг гарын сургалт
# ---------------------------------------------------------------------------

def train_one(hand: str) -> dict:
    if hand == "left":
        data_dir = os.path.join(BASE_DIR, "data", "left_dynamic")
        classes  = ["forward", "backward", "stop", "idle"]
        model_name = "left_transformer"
    else:
        data_dir = os.path.join(BASE_DIR, "data", "right_dynamic")
        classes  = ["left", "right", "idle"]
        model_name = "right_transformer"

    X, y = load_dataset(data_dir, classes)
    if len(X) == 0:
        print(f"[ERROR] {hand} гарын өгөгдөл олдсонгүй: {data_dir}")
        return {}

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_tmp, X_test, y_tmp, y_test, y_ti, y_testi = train_test_split(
        X, y_cat, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    X_train, X_val, y_train, y_val, y_tri, y_vali = train_test_split(
        X_tmp, y_tmp, y_ti, test_size=0.176, random_state=42, stratify=y_ti)

    print(f"\n[{hand.upper()}] Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

    model = build_transformer(len(encoder.classes_))
    model.summary()

    save_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True),
        ],
        verbose=1,
    )
    train_time = time.perf_counter() - t0

    np.save(os.path.join(MODEL_DIR, f"{model_name.replace('transformer','label_map')}.npy"),
            encoder.classes_)

    # Test үнэлгээ
    test_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_report = classification_report(y_testi, test_preds,
                                        target_names=encoder.classes_, digits=4)
    test_f1 = f1_score(y_testi, test_preds, average="macro")

    print(f"\n[{hand.upper()}] Test report:\n{test_report}")

    # Inference хурд хэмжих
    dummy = np.random.rand(1, SEQ_LEN, FEATURE_SIZE).astype(np.float32)
    for _ in range(5): model(dummy, training=False)
    t0 = time.perf_counter()
    for _ in range(200): model(dummy, training=False)
    inf_ms = (time.perf_counter() - t0) / 200 * 1000

    with open(os.path.join(METRIC_DIR, f"{model_name}_test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)

    # Training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(f"{hand} Transformer Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(f"{hand} Transformer Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(METRIC_DIR, f"{model_name}_curves.png"), dpi=120)
    plt.close()

    return {
        "hand":       hand,
        "test_f1":    test_f1,
        "inf_ms":     inf_ms,
        "train_time": train_time,
        "epochs":     len(history.history["loss"]),
        "test_report": test_report,
    }


# ---------------------------------------------------------------------------
# LSTM-тэй харьцуулах
# ---------------------------------------------------------------------------

def compare_with_lstm(results: list) -> None:
    """LSTM тест тайлан байвал Transformer-тэй харьцуулна."""
    lines = [
        "=" * 65,
        "Transformer vs LSTM — Харьцуулалт",
        "=" * 65,
        f"{'Загвар':<28} {'Test F1':>9} {'Inference':>11} {'Epoch':>7}",
        "-" * 65,
    ]

    for r in results:
        hand = r["hand"]

        # LSTM тест тайлан байвал F1 уншина
        lstm_report_path = os.path.join(METRIC_DIR, f"{hand}_test_report.txt")
        lstm_f1_str = "N/A"
        if os.path.exists(lstm_report_path):
            with open(lstm_report_path, encoding="utf-8") as f:
                for line in f:
                    if "macro avg" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            lstm_f1_str = parts[3]

        lines.append(f"LSTM ({hand}){'':<17} {lstm_f1_str:>9} {'—':>11} {'—':>7}")
        lines.append(
            f"Transformer ({hand}){'':<12} "
            f"{r['test_f1']:>9.4f} {r['inf_ms']:>9.2f}ms {r['epochs']:>7}"
        )
        lines.append("-" * 65)

    lines += [
        "",
        "Тайлбар:",
        "  Test F1     — macro-averaged F1 score (1.0 = төгс)",
        "  Inference   — нэг sequence prediction хугацаа",
        "  Epoch       — early stopping идэвхжсэн эцсийн epoch",
        "",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(METRIC_DIR, "transformer_comparison.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Харьцуулалт хадгалагдлаа: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    args = parser.parse_args()

    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    results = []
    for hand in hands:
        r = train_one(hand)
        if r:
            results.append(r)

    if results:
        compare_with_lstm(results)


if __name__ == "__main__":
    main()
