"""
latency_logger.py — Системийн latency хэмжиж CSV-д хадгална.
Туршилтын бүлэгт ашиглах тоон үзүүлэлт цуглуулах зорилготой.
"""
import time
import csv
import os
import logging
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


class LatencyLogger:
    """
    Gesture болон Voice pipeline-н latency-г хэмжиж хадгална.

    Хэрэглэх:
        ll = LatencyLogger("latency_log.csv")
        t = ll.start("gesture")
        ...боловсруулалт...
        ll.end("gesture", t, cmd="forward")
    """

    FIELDS = ["timestamp", "pipeline", "latency_ms", "command", "success"]

    def __init__(self, output_path: str = "metrics/latency_log.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.path = output_path
        self._file = open(output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

        # Rolling statistics
        self._stats: dict = {
            "gesture": deque(maxlen=500),
            "voice":   deque(maxlen=200),
            "llm":     deque(maxlen=200),
            "fusion":  deque(maxlen=500),
        }

    def start(self, pipeline: str) -> float:
        """Хэмжилт эхлүүлж эхлэлийн цаг буцаана."""
        return time.perf_counter()

    def end(
        self,
        pipeline: str,
        t_start: float,
        command: Optional[str] = None,
        success: bool = True,
    ) -> float:
        """Хэмжилт дуусгаж latency_ms буцаана."""
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline":  pipeline,
            "latency_ms": f"{latency_ms:.2f}",
            "command":   command or "",
            "success":   int(success),
        }
        self._writer.writerow(row)
        self._file.flush()

        if pipeline in self._stats:
            self._stats[pipeline].append(latency_ms)

        return latency_ms

    def summary(self) -> dict:
        """Pipeline тус бүрийн дундаж/min/max статистик буцаана."""
        result = {}
        for name, vals in self._stats.items():
            if not vals:
                continue
            lst = list(vals)
            result[name] = {
                "count": len(lst),
                "mean_ms": round(sum(lst) / len(lst), 2),
                "min_ms":  round(min(lst), 2),
                "max_ms":  round(max(lst), 2),
                "p95_ms":  round(sorted(lst)[int(len(lst) * 0.95)], 2),
            }
        return result

    def print_summary(self):
        s = self.summary()
        print("\n===== LATENCY SUMMARY =====")
        for pipe, stat in s.items():
            print(f"  {pipe:10s}: mean={stat['mean_ms']}ms  "
                  f"min={stat['min_ms']}ms  max={stat['max_ms']}ms  "
                  f"p95={stat['p95_ms']}ms  n={stat['count']}")
        print("===========================\n")

    def close(self):
        self.print_summary()
        self._file.close()