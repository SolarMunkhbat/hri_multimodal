import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "data", "right_dynamic")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASSES = ["left", "right", "idle"]
SEQ_LEN = 40
FEATURE_SIZE = 63

os.makedirs(MODEL_DIR, exist_ok=True)


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

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_cat,
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
            os.path.join(MODEL_DIR, "right_lstm.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    np.save(os.path.join(MODEL_DIR, "right_label_map.npy"), encoder.classes_)
    print("[INFO] Right model saved")
    print("[INFO] Labels:", encoder.classes_)


if __name__ == "__main__":
    main()