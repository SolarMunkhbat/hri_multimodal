"""
augment_sequences.py — Gesture sequence өгөгдлийг augment хийж dataset-г өргөтгөнө.

Augmentation аргууд:
  1. Gaussian noise    — бага зэрэг шуугиан нэмэх
  2. Time warping      — хугацааны дарааллыг сунгах/шахах
  3. Scale jitter      — хэмжээний жижиг өөрчлөлт
  4. Mirror (X-axis)   — гарыг толин тусгал хийх

Хэрэглэх:
    python -m training.augment_sequences --hand left --factor 3
    python -m training.augment_sequences --hand right --factor 3
"""

import os
import argparse
import numpy as np
from scipy.interpolate import interp1d

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_LEFT    = os.path.join(BASE_DIR, "data", "left_dynamic")
DATA_RIGHT   = os.path.join(BASE_DIR, "data", "right_dynamic")
SEQ_LEN      = 40
FEATURE_SIZE = 63


# ---------------------------------------------------------------------------
# Augmentation функцүүд
# ---------------------------------------------------------------------------

def add_gaussian_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Landmark координатад бага зэрэг шуугиан нэмнэ."""
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def time_warp(seq: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Sequence-н хугацааны хурдыг жигд бус байдлаар өөрчилнэ."""
    orig_steps = np.arange(SEQ_LEN)
    warp = np.cumsum(np.random.normal(1.0, sigma, SEQ_LEN))
    warp = warp / warp[-1] * (SEQ_LEN - 1)
    warped = np.zeros_like(seq)
    for f in range(FEATURE_SIZE):
        interp = interp1d(orig_steps, seq[:, f], kind="linear", fill_value="extrapolate")
        warped[:, f] = interp(warp).astype(np.float32)
    return warped


def scale_jitter(seq: np.ndarray, scale_range=(0.85, 1.15)) -> np.ndarray:
    """Бүх coordinate-г тогтмол хүчин зүйлээр үржүүлнэ."""
    scale = np.random.uniform(*scale_range)
    return (seq * scale).astype(np.float32)


def mirror_x(seq: np.ndarray) -> np.ndarray:
    """X координатыг тусгана (гарыг толин тусгал)."""
    mirrored = seq.copy()
    # x coordinate нь 0, 3, 6, ... индексүүд (feature бүр x,y,z)
    mirrored[:, 0::3] = -mirrored[:, 0::3]
    return mirrored.astype(np.float32)


AUGMENTERS = [
    ("noise",    lambda s: add_gaussian_noise(s, sigma=0.005)),
    ("noise2",   lambda s: add_gaussian_noise(s, sigma=0.010)),
    ("warp",     lambda s: time_warp(s, sigma=0.10)),
    ("warp2",    lambda s: time_warp(s, sigma=0.20)),
    ("scale",    lambda s: scale_jitter(s, (0.90, 1.10))),
    ("mirror",   mirror_x),
    ("noise_warp", lambda s: time_warp(add_gaussian_noise(s), sigma=0.10)),
    ("scale_noise", lambda s: add_gaussian_noise(scale_jitter(s), sigma=0.005)),
]


# ---------------------------------------------------------------------------
# Augment хийх үндсэн функц
# ---------------------------------------------------------------------------

def augment_class_dir(class_dir: str, factor: int):
    """
    Нэг ангиллын хавтасны sequence файлуудыг augment хийнэ.
    factor: нэг sequence-с хэдэн augmented хувилбар үүсгэх
    """
    files = [f for f in os.listdir(class_dir) if f.endswith(".npy") and not f.startswith("aug_")]
    if not files:
        print(f"  [SKIP] {class_dir} — файл байхгүй")
        return 0

    created = 0
    augmenters = AUGMENTERS[:factor]

    for fname in files:
        path = os.path.join(class_dir, fname)
        seq = np.load(path)
        if seq.shape != (SEQ_LEN, FEATURE_SIZE):
            continue

        base = os.path.splitext(fname)[0]
        for aug_name, aug_fn in augmenters:
            out_name = f"aug_{base}_{aug_name}.npy"
            out_path = os.path.join(class_dir, out_name)
            if os.path.exists(out_path):
                continue
            aug_seq = aug_fn(seq)
            np.save(out_path, aug_seq)
            created += 1

    return created


def augment_dataset(data_dir: str, factor: int):
    print(f"\n[AUGMENT] {data_dir}  (factor={factor})")
    total = 0
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        orig = len([f for f in os.listdir(class_dir) if f.endswith(".npy") and not f.startswith("aug_")])
        n = augment_class_dir(class_dir, factor)
        aug_total = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
        print(f"  {class_name:15s}: {orig} → {aug_total} sequence (+{n} augmented)")
        total += n
    print(f"  Нийт нэмэгдсэн: {total} sequence\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture sequence augmentation")
    parser.add_argument("--hand",   choices=["left", "right", "both"], default="both")
    parser.add_argument("--factor", type=int, default=4,
                        help="Нэг sequence-с үүсгэх augmented хувилбарын тоо (1-8)")
    args = parser.parse_args()

    factor = max(1, min(8, args.factor))

    if args.hand in ("left", "both"):
        augment_dataset(DATA_LEFT, factor)
    if args.hand in ("right", "both"):
        augment_dataset(DATA_RIGHT, factor)