"""
convert_tflite.py — Keras загваруудыг TFLite болгон хөрвүүлж,
inference хурдыг benchmark хийнэ.

Ажиллуулах:
    python -m training.convert_tflite

Гаралт:
    models/left_lstm.tflite
    models/right_lstm.tflite
    metrics/tflite_benchmark.txt
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
METRIC_DIR = os.path.join(BASE_DIR, "metrics")
SEQ_LEN    = 40
FEATURE_SIZE = 63
N_BENCH    = 200   # benchmark давтлагын тоо

os.makedirs(METRIC_DIR, exist_ok=True)


def _build_lstm(num_classes: int):
    """Training-тай яг адил архитектур — load_model алдаатай үед ашиглана."""
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(SEQ_LEN, FEATURE_SIZE)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def _load_model(keras_path: str, labels_path: str):
    """Keras version mismatch тохиолдолд weights-г шуудхэн ачаална."""
    try:
        return tf.keras.models.load_model(keras_path, compile=False)
    except Exception:
        labels = np.load(labels_path, allow_pickle=True)
        model = _build_lstm(len(labels))
        model.load_weights(keras_path)
        return model


def convert(keras_path: str, tflite_path: str, labels_path: str) -> None:
    print(f"[INFO] Хөрвүүлж байна: {keras_path}")
    model = _load_model(keras_path, labels_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # LSTM нь TensorListReserve op ашигладаг тул SELECT_TF_OPS шаардлагатай
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"[INFO] Хадгалагдлаа: {tflite_path}  ({size_kb:.1f} KB)")


def benchmark_keras(keras_path: str, labels_path: str, n: int) -> float:
    model = _load_model(keras_path, labels_path)
    dummy = np.random.rand(1, SEQ_LEN, FEATURE_SIZE).astype(np.float32)
    # warm-up
    for _ in range(5):
        model(dummy, training=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model(dummy, training=False)
    return (time.perf_counter() - t0) / n * 1000   # ms/inference


def benchmark_tflite(tflite_path: str, n: int) -> float:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    dummy = np.random.rand(1, SEQ_LEN, FEATURE_SIZE).astype(np.float32)
    # warm-up
    for _ in range(5):
        interp.set_tensor(inp_idx, dummy)
        interp.invoke()
    t0 = time.perf_counter()
    for _ in range(n):
        interp.set_tensor(inp_idx, dummy)
        interp.invoke()
    return (time.perf_counter() - t0) / n * 1000   # ms/inference


def main():
    pairs = [
        ("left_lstm.keras",  "left_lstm.tflite",  "left_label_map.npy"),
        ("right_lstm.keras", "right_lstm.tflite", "right_label_map.npy"),
    ]

    lines = [
        "=" * 55,
        "TFLite Conversion Benchmark",
        "=" * 55,
        f"{'Загвар':<20} {'Keras (ms)':>12} {'TFLite (ms)':>12} {'Хурдан':>8}",
        "-" * 55,
    ]

    for keras_name, tflite_name, labels_name in pairs:
        kpath = os.path.join(MODEL_DIR, keras_name)
        tpath = os.path.join(MODEL_DIR, tflite_name)
        lpath = os.path.join(MODEL_DIR, labels_name)

        if not os.path.exists(kpath):
            print(f"[WARN] Загвар олдсонгүй: {kpath}")
            continue

        convert(kpath, tpath, lpath)

        k_ms = benchmark_keras(kpath, lpath, N_BENCH)
        t_ms = benchmark_tflite(tpath, N_BENCH)
        speedup = k_ms / t_ms if t_ms > 0 else 0

        lines.append(
            f"{keras_name:<20} {k_ms:>12.3f} {t_ms:>12.3f} {speedup:>7.2f}×"
        )
        print(f"  Keras: {k_ms:.3f}ms  TFLite: {t_ms:.3f}ms  → {speedup:.2f}× хурдан")

    lines.append("")
    report = "\n".join(lines)
    print("\n" + report)

    out = os.path.join(METRIC_DIR, "tflite_benchmark.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Benchmark хадгалагдлаа: {out}")


if __name__ == "__main__":
    main()
