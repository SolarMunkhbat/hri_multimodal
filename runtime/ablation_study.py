"""
ablation_study.py — Gesture-only / Voice-only / Multimodal горимуудыг харьцуулна.

Ажиллуулах:
    python -m runtime.ablation_study --log metrics/latency_log.csv

Гаралт:
    metrics/ablation_report.txt
    metrics/ablation_chart.png
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

# Windows terminal UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRIC_DIR = "metrics"


# ---------------------------------------------------------------------------
# Latency log уншина
# ---------------------------------------------------------------------------

def load_latency_log(path: str) -> Dict[str, List[float]]:
    """CSV log-г уншиж channel-р ангилна."""
    data: Dict[str, List[float]] = defaultdict(list)
    if not os.path.exists(path):
        print(f"[WARN] Log файл олдсонгүй: {path}")
        return data
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch  = row.get("channel", "").strip()
            try:
                ms = float(row.get("latency_ms", 0))
            except ValueError:
                continue
            if ch and ms > 0:
                data[ch].append(ms)
    return data


# ---------------------------------------------------------------------------
# Горим тус бүрийн онолын latency тооцоо
# ---------------------------------------------------------------------------

def compute_mode_stats(latency: Dict[str, List[float]]) -> Dict[str, dict]:
    """
    Гурван горимын latency-г log өгөгдлөөс тооцооллоно.

    Gesture-only  : gesture latency
    Voice-only    : voice/llm latency (хэрэв байвал)
    Multimodal    : gesture + fusion latency
    """
    def stats(values: List[float]) -> dict:
        if not values:
            return {"mean": 0, "p95": 0, "min": 0, "max": 0, "n": 0}
        a = np.array(values)
        return {
            "mean": float(np.mean(a)),
            "p95":  float(np.percentile(a, 95)),
            "min":  float(np.min(a)),
            "max":  float(np.max(a)),
            "n":    len(a),
        }

    g  = latency.get("gesture", [])
    f  = latency.get("fusion",  [])
    v  = latency.get("voice",   latency.get("llm", []))

    # Multimodal = gesture + fusion (хамгийн урт замын latency)
    mm_combined = [
        a + b for a, b in zip(g[:len(f)], f[:len(g)])
    ] if g and f else g or f

    return {
        "Gesture-only":  stats(g),
        "Voice-only":    stats(v),
        "Multimodal":    stats(mm_combined),
    }


# ---------------------------------------------------------------------------
# Дүрслэл
# ---------------------------------------------------------------------------

def plot_ablation(mode_stats: Dict[str, dict], out_path: str) -> None:
    modes  = list(mode_stats.keys())
    means  = [mode_stats[m]["mean"] for m in modes]
    p95s   = [mode_stats[m]["p95"]  for m in modes]
    errors = [p - mu for mu, p in zip(means, p95s)]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, means,  width, label="Дундаж (ms)",  color="#4C72B0")
    bars2 = ax.bar(x + width/2, p95s,   width, label="P95 (ms)",     color="#DD8452")

    ax.set_xlabel("Горим")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Ablation Study — Горим тус бүрийн latency харьцуулалт")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] График хадгалагдлаа: {out_path}")


# ---------------------------------------------------------------------------
# Тайлан
# ---------------------------------------------------------------------------

def write_report(mode_stats: Dict[str, dict], latency: Dict[str, List[float]],
                 out_path: str) -> None:
    lines = [
        "=" * 60,
        "ABLATION STUDY — HRI Multimodal Control System",
        "=" * 60,
        "",
        f"{'Горим':<18} {'Дундаж (ms)':>12} {'P95 (ms)':>10} {'Min':>8} {'Max':>8} {'N':>6}",
        "-" * 60,
    ]
    for mode, s in mode_stats.items():
        lines.append(
            f"{mode:<18} {s['mean']:>12.2f} {s['p95']:>10.2f} "
            f"{s['min']:>8.2f} {s['max']:>8.2f} {s['n']:>6}"
        )

    lines += [
        "",
        "Тайлбар:",
        "  Gesture-only  — зөвхөн гарын дохио gesture latency хэмждэг",
        "  Voice-only    — зөвхөн дуу хоолойн тушаал STT+parse latency",
        "  Multimodal    — gesture + fusion хосолсон горим (энэ систем)",
        "",
        "Дүгнэлт:",
    ]

    g_mean = mode_stats["Gesture-only"]["mean"]
    v_mean = mode_stats["Voice-only"]["mean"]
    m_mean = mode_stats["Multimodal"]["mean"]

    if g_mean > 0 and v_mean > 0:
        lines.append(
            f"  Multimodal горим нь gesture-only-с {m_mean/g_mean:.2f}× "
            f"latency-тэй боловч хоёр модальтийг нэгтгэдэг."
        )
    if v_mean > 0 and m_mean > 0 and m_mean < v_mean:
        lines.append(
            "  Multimodal нь voice-only-с хурдан учир real-time удирдлагад "
            "gesture-г голлон ашигладаг."
        )
    lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Тайлан хадгалагдлаа: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HRI Ablation Study")
    parser.add_argument("--log", default="metrics/latency_log.csv",
                        help="Latency log CSV файл")
    args = parser.parse_args()

    os.makedirs(METRIC_DIR, exist_ok=True)

    latency    = load_latency_log(args.log)
    mode_stats = compute_mode_stats(latency)

    plot_ablation(mode_stats, os.path.join(METRIC_DIR, "ablation_chart.png"))
    write_report(mode_stats, latency, os.path.join(METRIC_DIR, "ablation_report.txt"))


if __name__ == "__main__":
    main()
