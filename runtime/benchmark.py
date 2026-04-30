"""
benchmark.py — LLM vs Keyword командын нарийвчлал болон хурдыг харьцуулна.

Туршилтын бүлэгт ашиглах тоон үзүүлэлт гаргах зорилготой.

Хэрэглэх:
    python -m runtime.benchmark
"""
import time
import logging
import csv
import os

logging.basicConfig(level=logging.WARNING)

from runtime.llm_control import OllamaCommandParser, _KeywordFallback

OUTPUT_DIR = "metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Туршилтын dataset — (оролт текст, хүлээгдэж буй команд)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Стандарт монгол командууд ---
    ("урагш яв",                    "forward"),
    ("урагшаа",                     "forward"),
    ("урагшаа явна уу",             "forward"),
    ("зогс",                        "stop"),
    ("зогсоо",                      "stop"),
    ("машиныг зогсоо",              "stop"),
    ("баруун тийш эргэ",            "right"),
    ("баруун эрэг",                 "right"),
    ("зүүн тийш эргэ",              "left"),
    ("зүүн эрэг",                   "left"),
    ("ухар",                        "backward"),
    ("ухраа",                       "backward"),
    ("арагш яв",                    "backward"),
    ("хурдаа нэм",                  "faster"),
    ("хурдан явна уу",              "faster"),
    ("удаан яв",                    "slower"),
    ("аяар яв",                     "slower"),
    ("үргэлжлүүл",                  "resume"),
    ("цааш яв",                     "resume"),
    ("эргэхээ боль",                "turn_stop"),
    ("чигээрээ яваарай",            "turn_stop"),
    ("явахаа боль",                 "move_stop"),
    ("90 градус зүүн эргэ",         "turn_left_90"),
    ("ерэн градус баруун эргэ",     "turn_right_90"),
    ("баруун тийш 45 градус эргэ",  "turn_right_45"),
    # --- Байгалийн хэллэг (keyword-д байхгүй) ---
    ("нааш ир",                     "forward"),
    ("аяар урагшаа яв",             "forward"),
    ("удаан урагшаа яв",            "forward"),
    ("хурдан баруун эрэг",          "right"),
    ("бага зэрэг зүүн тийш",        "left"),
    # --- Англи ---
    ("go forward",                  "forward"),
    ("stop the robot",              "stop"),
    ("turn right",                  "right"),
    ("speed up",                    "faster"),
    ("slow down",                   "slower"),
    # --- Командгүй ---
    ("сайн байна уу",               None),
    ("цаг ямар байна",              None),
    ("тэгээрэй",                    None),
]


def run_benchmark(parser, name: str, test_cases: list) -> dict:
    results = []
    correct = 0
    total_time = 0.0

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    for text, expected in test_cases:
        t0 = time.perf_counter()
        got = parser.parse(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ok = (got == expected)
        if ok:
            correct += 1
        total_time += elapsed_ms

        status = "✓" if ok else "✗"
        print(f"  {status}  {text!r:35s} → {str(got):18s}  [{elapsed_ms:.0f}ms]")
        results.append({
            "method":    name,
            "input":     text,
            "expected":  str(expected),
            "got":       str(got),
            "correct":   int(ok),
            "latency_ms": round(elapsed_ms, 2),
        })

    accuracy = correct / len(test_cases) * 100
    avg_ms   = total_time / len(test_cases)
    print(f"\n  Нарийвчлал: {correct}/{len(test_cases)} = {accuracy:.1f}%")
    print(f"  Дундаж хугацаа: {avg_ms:.1f}ms")

    return {
        "name":     name,
        "correct":  correct,
        "total":    len(test_cases),
        "accuracy": round(accuracy, 1),
        "avg_ms":   round(avg_ms, 1),
        "rows":     results,
    }


def save_csv(all_rows: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method","input","expected","got","correct","latency_ms"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[CSV] {path} хадгалагдлаа")


def print_comparison_table(results: list):
    print("\n" + "="*55)
    print("  ХАРЬЦУУЛАЛТЫН ХҮСНЭГТ")
    print("="*55)
    print(f"  {'Арга':20s}  {'Нарийвчлал':>12s}  {'Дундаж хугацаа':>15s}")
    print("-"*55)
    for r in results:
        print(f"  {r['name']:20s}  {r['accuracy']:>10.1f}%  {r['avg_ms']:>12.1f}ms")
    print("="*55)


if __name__ == "__main__":
    all_rows = []
    summary  = []

    # 1. Keyword fallback
    keyword = _KeywordFallback()
    r1 = run_benchmark(keyword, "Keyword Fallback", TEST_CASES)
    all_rows.extend(r1["rows"])
    summary.append(r1)

    # 2. Ollama LLM
    print("\n[LLM] Ollama холбогдож байна...")
    llm = OllamaCommandParser(model="llama3.2")
    if llm.health_check():
        r2 = run_benchmark(llm, "Ollama LLM (llama3.2)", TEST_CASES)
        all_rows.extend(r2["rows"])
        summary.append(r2)
    else:
        print("[LLM] Ollama ажиллахгүй байна — зөвхөн keyword тест хийгдлээ")

    # Хадгалах
    save_csv(all_rows, os.path.join(OUTPUT_DIR, "benchmark_results.csv"))
    print_comparison_table(summary)