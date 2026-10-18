"""Тайм-бюджет стадии: каждый тяжёлый шаг спрашивает остаток и умеет
остановиться на полпути. Резерв на запись файла неприкосновенен."""
import time

from src.config import SAFETY_FRACTION, WRITE_RESERVE_SEC, stage_time_limit


class TimeBudget:
    def __init__(self, n_rows: int):
        self.start = time.monotonic()
        self.limit = stage_time_limit(n_rows) * SAFETY_FRACTION

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return max(0.0, self.limit - self.elapsed() - WRITE_RESERVE_SEC)

    def has(self, seconds: float) -> bool:
        return self.remaining() >= seconds

    def log(self, tag: str):
        print(f"[budget] {tag}: elapsed={self.elapsed():.0f}s remaining={self.remaining():.0f}s", flush=True)
