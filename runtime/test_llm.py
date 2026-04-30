"""
test_llm.py — Ollama LLM командын парсерыг тест хийнэ.

Ажиллуулах:
    python -m runtime.test_llm
"""
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

from runtime.llm_control import OllamaCommandParser

parser = OllamaCommandParser(model="llama3.2")

print(f"\nOllama боломжтой эсэх: {parser.health_check()}\n")

test_cases = [
    # Монгол
    ("урагш яв",              "forward"),
    ("зогс",                  "stop"),
    ("баруун тийш эргэ",      "right"),
    ("зүүн тийш эргэ",             "left"),
    ("ухар",                 "backward"),
    ("хурдаа нэм",           "faster"),
    ("удаан яв",              "slower"),
    ("90 градус зүүн эргэ",   "turn_left_90"),
    ("баруун тийш 45 градус эргэ", "turn_right_45"),
    ("үргэлжлүүл",            "resume"),
    ("эргэхээ боль",          "turn_stop"),
    # Англи
    ("go forward",            "forward"),
    ("stop the robot",        "stop"),
    ("turn right",            "right"),
    ("speed up",              "faster"),
    # Тохирохгүй
    ("сайн байна уу",         None),
    ("цаг ямар байна",        None),
]

passed = 0
for text, expected in test_cases:
    result = parser.parse(text)
    ok = result == expected
    status = "✓" if ok else "✗"
    if ok:
        passed += 1
    print(f"  {status}  {text!r:35s} → {str(result):15s}  (expected: {str(expected)})")

print(f"\n{passed}/{len(test_cases)} тест амжилттай\n")