"""Mapping between Python AST node positions and offsets in the source text.

``ast`` reports ``col_offset`` as a *byte* offset into the UTF-8 encoding of the
line, not a character offset, so a naive ``line[col_offset:]`` slice silently
corrupts any file containing non-ASCII text.  :class:`SourceMap` does the
conversion properly and is the single place in the tool that knows about it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True, order=True)
class Position:
    """A location in a source file. ``line`` is 1-based, ``column`` 0-based."""

    line: int
    column: int

    def __str__(self) -> str:
        return "{}:{}".format(self.line, self.column)

    def as_dict(self) -> dict:
        return {"line": self.line, "column": self.column}

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(line=int(data["line"]), column=int(data["column"]))


class SourceMap:
    """Converts ``ast`` node positions into character offsets in a source string."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines: List[str] = source.splitlines(keepends=True)
        line_start: List[int] = []
        running = 0
        for line in self.lines:
            line_start.append(running)
            running += len(line)
        line_start.append(running)
        self._line_start = line_start

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def char_column(self, line: int, byte_column: int) -> int:
        """Translate a byte column reported by ``ast`` into a character column."""
        if byte_column <= 0:
            return 0
        if line < 1 or line > len(self.lines):
            return byte_column
        prefix = self.lines[line - 1].encode("utf-8")[:byte_column]
        return len(prefix.decode("utf-8", errors="ignore"))

    def offset(self, position: Position) -> int:
        """Character offset of ``position`` within the source string."""
        line = min(max(position.line, 1), len(self._line_start))
        start = self._line_start[line - 1]
        limit = self._line_start[line] if line < len(self._line_start) else len(self.source)
        return min(start + max(position.column, 0), limit)

    def span(self, node: ast.AST) -> Optional[Tuple[Position, Position]]:
        """Start/end :class:`Position` of ``node``, or ``None`` if unavailable.

        Nodes synthesised by the parser (and, before Python 3.12, everything
        nested inside an f-string) may carry no or bogus end positions; callers
        must treat ``None`` as "not a mutable location".
        """
        start_line = getattr(node, "lineno", None)
        start_col = getattr(node, "col_offset", None)
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if start_line is None or start_col is None or end_line is None or end_col is None:
            return None
        start = Position(start_line, self.char_column(start_line, start_col))
        end = Position(end_line, self.char_column(end_line, end_col))
        if end <= start:
            return None
        return start, end

    def text(self, start: Position, end: Position) -> str:
        return self.source[self.offset(start) : self.offset(end)]

    def replace(self, start: Position, end: Position, replacement: str) -> str:
        """Return the source with the ``start``..``end`` span swapped out."""
        return self.source[: self.offset(start)] + replacement + self.source[self.offset(end) :]

    def window(self, start: Position, end: Position, max_lines: int) -> Tuple[int, int]:
        """Pick a line range of at most ``max_lines`` lines around a span.

        Returns 1-based inclusive ``(first_line, last_line)``.  Mirrors the
        paper's default of including 200 lines of context around a placeholder.
        """
        total = max(len(self.lines), 1)
        if max_lines <= 0 or total <= max_lines:
            return 1, total
        span_lines = end.line - start.line + 1
        if span_lines >= max_lines:
            return start.line, min(total, start.line + max_lines - 1)
        padding = (max_lines - span_lines) // 2
        first = max(1, start.line - padding)
        last = min(total, first + max_lines - 1)
        first = max(1, last - max_lines + 1)
        return first, last

    def line_text(self, first: int, last: int) -> str:
        return "".join(self.lines[first - 1 : last])
