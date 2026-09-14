"""Minimal console output helpers (no third-party dependencies)."""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from typing import Optional, TextIO


@dataclass
class Console:
    stream: TextIO = sys.stderr
    quiet: bool = False
    use_colour: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.use_colour is None:
            self.use_colour = bool(getattr(self.stream, "isatty", lambda: False)())

    def _paint(self, text: str, code: str) -> str:
        if not self.use_colour:
            return text
        return "\033[{}m{}\033[0m".format(code, text)

    def info(self, message: str = "") -> None:
        if not self.quiet:
            print(message, file=self.stream, flush=True)

    def step(self, message: str) -> None:
        self.info(self._paint("==> ", "1;36") + message)

    def warn(self, message: str) -> None:
        print(self._paint("warning: ", "1;33") + message, file=self.stream, flush=True)

    def error(self, message: str) -> None:
        print(self._paint("error: ", "1;31") + message, file=self.stream, flush=True)

    def progress(self, done: int, total: int, suffix: str = "") -> None:
        """Overwrite a single status line; falls back to nothing when quiet."""
        if self.quiet or total <= 0:
            return
        width = max(20, min(shutil.get_terminal_size((80, 20)).columns - 30, 40))
        filled = int(width * done / total)
        bar = "#" * filled + "." * (width - filled)
        line = "\r[{}] {}/{} {}".format(bar, done, total, suffix)
        if getattr(self.stream, "isatty", lambda: False)():
            self.stream.write(line[: shutil.get_terminal_size((120, 20)).columns - 1])
            self.stream.flush()
            if done >= total:
                self.stream.write("\n")
                self.stream.flush()
        elif done >= total or done % 50 == 0:
            print(line.strip(), file=self.stream, flush=True)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "{:.0f}ms".format(seconds * 1000)
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return "{:.0f}m{:02.0f}s".format(minutes, rest)
    hours, minutes = divmod(minutes, 60)
    return "{:.0f}h{:02.0f}m".format(hours, minutes)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.monotonic() - self.start

    @property
    def seconds(self) -> float:
        return getattr(self, "elapsed", time.monotonic() - self.start)
