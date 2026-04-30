import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "data", "left_dynamic")
MODEL_DIR = os.path.join(BASE_DIR, "models")
METRIC_DIR = os.path.join(BASE_DIR, "metrics")

CLASSES = ["forward", "backward", "stop", "idle"]
SEQ_LEN = 40
FEATURE_SIZE = 63

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)


def load_dataset():
    X = []
    y = []

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            if not file_name.endswith(".npy"):
                continue

            path = os.path.join(class_dir, file_name)
            seq = np.load(path)

            if seq.shape != (SEQ_LEN, FEATURE_SIZE):
                print(f"[WARN] skipped {path}, shape={seq.shape}")
                continue

            X.append(seq)
            y.append(class_name)

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, FEATURE_SIZE)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_history_plots(history, prefix):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(METRIC_DIR, f"{prefix}_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(METRIC_DIR, f"{prefix}_accuracy.png"))
    plt.close()


def save_confusion_matrix(cm, class_names, prefix):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{prefix} Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(METRIC_DIR, f"{prefix}_confusion_matrix.png"))
    plt.close()


def main():
    X, y = load_dataset()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) == 0:
        print("[ERROR] No training data found")
        return

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_val, y_train, y_val, y_train_idx, y_val_idx = train_test_split(
        X,
        y_cat,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = build_model(len(encoder.classes_))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "left_lstm.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    np.save(os.path.join(MODEL_DIR, "left_label_map.npy"), encoder.classes_)

    val_pred_prob = model.predict(X_val, verbose=0)
    val_pred_idx = np.argmax(val_pred_prob, axis=1)

    report = classification_report(
        y_val_idx,
        val_pred_idx,
        target_names=encoder.classes_,
        digits=4
    )
    cm = confusion_matrix(y_val_idx, val_pred_idx)

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    with open(os.path.join(METRIC_DIR, "left_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    np.save(os.path.join(METRIC_DIR, "left_confusion_matrix.npy"), cm)

    save_history_plots(history, "left")
    save_confusion_matrix(cm, encoder.classes_, "left")

    print("[INFO] Left model saved")
    print("[INFO] Labels:", encoder.classes_)
    print("[INFO] Metrics saved in metrics/ folder")


if __name__ == "__main__":
    main()