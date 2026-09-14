"""Deciding *where* in a Python file a placeholder should be introduced.

This is the Python counterpart of the placeholder scheme in Figure 6 of the
LLMorpheus paper.  The paper targets (i) the conditions of ``if``/``switch``/
``while``/``do-while``, (ii) the initialiser, updater and full header of loops,
and (iii) the receiver, individual arguments and full argument list of calls.

The default scheme below is the closest Python analogue.  As in the paper, the
point is to strike a balance between producing a practical number of mutants and
targeting places where a change is likely to alter control or data flow.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .positions import Position, SourceMap

# --- Placeholder kinds -----------------------------------------------------
# Mirroring the paper's scheme.
IF_CONDITION = "if-condition"
WHILE_CONDITION = "while-condition"
TERNARY_CONDITION = "ternary-condition"
ASSERT_CONDITION = "assert-condition"
MATCH_SUBJECT = "match-subject"
FOR_TARGET = "for-target"
FOR_ITER = "for-iter"
COMPREHENSION_ITER = "comprehension-iter"
COMPREHENSION_CONDITION = "comprehension-condition"
CALL_CALLEE = "call-callee"
CALL_ARGUMENT = "call-argument"
CALL_KEYWORD_ARGUMENT = "call-keyword-argument"
CALL_ARGUMENTS = "call-arguments"

# Beyond the paper: useful, but they widen the mutant count noticeably.
RETURN_VALUE = "return-value"
ASSIGN_VALUE = "assign-value"
SUBSCRIPT_INDEX = "subscript-index"
WITH_CONTEXT = "with-context"
EXCEPT_TYPE = "except-type"

DEFAULT_KINDS: Tuple[str, ...] = (
    IF_CONDITION,
    WHILE_CONDITION,
    TERNARY_CONDITION,
    ASSERT_CONDITION,
    MATCH_SUBJECT,
    FOR_TARGET,
    FOR_ITER,
    COMPREHENSION_ITER,
    COMPREHENSION_CONDITION,
    CALL_CALLEE,
    CALL_ARGUMENT,
    CALL_KEYWORD_ARGUMENT,
    CALL_ARGUMENTS,
)

EXTENDED_KINDS: Tuple[str, ...] = DEFAULT_KINDS + (
    RETURN_VALUE,
    ASSIGN_VALUE,
    SUBSCRIPT_INDEX,
    WITH_CONTEXT,
    EXCEPT_TYPE,
)

ALL_KINDS: Tuple[str, ...] = EXTENDED_KINDS


@dataclass(frozen=True)
class Location:
    """A single span of source code that will be replaced by a placeholder."""

    id: str
    file: str  # path relative to the project root, POSIX separators
    kind: str
    start: Position
    end: Position
    original: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "kind": self.kind,
            "start": self.start.as_dict(),
            "end": self.end.as_dict(),
            "original": self.original,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Location":
        return cls(
            id=data["id"],
            file=data["file"],
            kind=data["kind"],
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
            original=data["original"],
        )

    def describe(self) -> str:
        return "{}:{} [{}] {}".format(self.file, self.start, self.kind, _one_line(self.original))


def _one_line(text: str, limit: int = 72) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _location_id(file: str, kind: str, start: Position, end: Position) -> str:
    digest = hashlib.sha1(
        "{}|{}|{}|{}".format(file, kind, start, end).encode("utf-8")
    ).hexdigest()[:10]
    return "{}#{}".format(digest, kind)


class _LocationCollector(ast.NodeVisitor):
    """Walks a module and records every span eligible for a placeholder."""

    def __init__(
        self,
        file: str,
        source_map: SourceMap,
        kinds: Set[str],
        max_fragment_chars: int,
        max_fragment_lines: int,
    ) -> None:
        self.file = file
        self.smap = source_map
        self.kinds = kinds
        self.max_fragment_chars = max_fragment_chars
        self.max_fragment_lines = max_fragment_lines
        self.locations: List[Location] = []
        self._seen: Set[Tuple[str, int, int, int, int]] = set()

    # -- recording ---------------------------------------------------------
    def _add_node(self, kind: str, node: Optional[ast.AST]) -> None:
        if node is None or kind not in self.kinds:
            return
        span = self.smap.span(node)
        if span is None:
            return
        self._add_span(kind, span[0], span[1])

    def _add_span(self, kind: str, start: Position, end: Position) -> None:
        if kind not in self.kinds:
            return
        key = (kind, start.line, start.column, end.line, end.column)
        if key in self._seen:
            return
        original = self.smap.text(start, end)
        if not original.strip():
            return
        if len(original) > self.max_fragment_chars:
            return
        if end.line - start.line + 1 > self.max_fragment_lines:
            return
        self._seen.add(key)
        self.locations.append(
            Location(
                id=_location_id(self.file, kind, start, end),
                file=self.file,
                kind=kind,
                start=start,
                end=end,
                original=original,
            )
        )

    # -- traversal ---------------------------------------------------------
    def visit_JoinedStr(self, node: ast.AST) -> None:
        """Treat f-strings as opaque leaves.

        Before Python 3.12 the nodes nested inside an f-string carry positions
        that do not correspond to the real source text, so descending into them
        would produce spans that silently mangle the file.
        """
        return

    def visit_If(self, node: ast.If) -> None:
        self._add_node(IF_CONDITION, node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._add_node(TERNARY_CONDITION, node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add_node(WHILE_CONDITION, node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._add_node(ASSERT_CONDITION, node.test)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add_node(FOR_TARGET, node.target)
        self._add_node(FOR_ITER, node.iter)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._add_node(COMPREHENSION_ITER, node.iter)
        for condition in node.ifs:
            self._add_node(COMPREHENSION_CONDITION, condition)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._add_node(CALL_CALLEE, node.func)
        for argument in node.args:
            self._add_node(CALL_ARGUMENT, argument)
        for keyword in node.keywords:
            self._add_node(CALL_KEYWORD_ARGUMENT, keyword.value)
        self._add_argument_list(node)
        self.generic_visit(node)

    def _add_argument_list(self, node: ast.Call) -> None:
        """Record the full argument list as one span, e.g. ``f(<PLACEHOLDER>)``."""
        if CALL_ARGUMENTS not in self.kinds:
            return
        parts: List[ast.AST] = list(node.args)
        # ``ast.keyword`` carries its own position since Python 3.9, which is
        # what we want: it covers the ``name=`` prefix and the ``**`` of
        # ``**kwargs``.  Fall back to the value for older/synthesised nodes.
        for keyword in node.keywords:
            parts.append(keyword if getattr(keyword, "lineno", None) else keyword.value)
        spans = [self.smap.span(part) for part in parts]
        spans = [span for span in spans if span is not None]
        if len(spans) < 1:
            return
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        # A single positional argument is already covered by CALL_ARGUMENT.
        if len(spans) == 1 and len(node.args) == 1:
            return
        self._add_span(CALL_ARGUMENTS, start, end)

    # -- beyond the paper --------------------------------------------------
    def visit_Return(self, node: ast.Return) -> None:
        self._add_node(RETURN_VALUE, node.value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._add_node(ASSIGN_VALUE, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_node(ASSIGN_VALUE, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_node(ASSIGN_VALUE, node.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        target = node.slice
        # Python 3.8 wrapped simple subscripts in ast.Index; unwrap it.
        target = getattr(target, "value", target) if target.__class__.__name__ == "Index" else target
        if isinstance(target, ast.AST) and not isinstance(target, ast.Slice):
            self._add_node(SUBSCRIPT_INDEX, target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self._add_node(WITH_CONTEXT, item.context_expr)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._add_node(EXCEPT_TYPE, node.type)
        self.generic_visit(node)


def _visit_match(collector: _LocationCollector, node: ast.AST) -> None:
    collector._add_node(MATCH_SUBJECT, getattr(node, "subject", None))
    collector.generic_visit(node)


if hasattr(ast, "Match"):  # Python 3.10+
    setattr(_LocationCollector, "visit_Match", _visit_match)


def resolve_kinds(selected: Optional[Iterable[str]]) -> Set[str]:
    """Turn a user-supplied ``--kinds`` selection into a concrete set."""
    if not selected:
        return set(DEFAULT_KINDS)
    resolved: Set[str] = set()
    for item in selected:
        name = item.strip()
        if not name:
            continue
        if name in ("default", "paper"):
            resolved.update(DEFAULT_KINDS)
        elif name in ("all", "extended"):
            resolved.update(EXTENDED_KINDS)
        elif name in ALL_KINDS:
            resolved.add(name)
        else:
            raise ValueError(
                "unknown placeholder kind {!r}; known kinds: {}".format(
                    name, ", ".join(ALL_KINDS)
                )
            )
    return resolved


def find_locations(
    source: str,
    file: str,
    kinds: Optional[Iterable[str]] = None,
    max_fragment_chars: int = 240,
    max_fragment_lines: int = 8,
) -> List[Location]:
    """Return every placeholder location in ``source``.

    ``file`` is the project-relative path used in mutant identifiers.  A file
    that does not parse yields no locations rather than raising, so a project
    with one unparsable module can still be mutated.
    """
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError:
        return []
    collector = _LocationCollector(
        file=file,
        source_map=SourceMap(source),
        kinds=resolve_kinds(kinds),
        max_fragment_chars=max_fragment_chars,
        max_fragment_lines=max_fragment_lines,
    )
    collector.visit(tree)
    collector.locations.sort(key=lambda loc: (loc.start, loc.end, loc.kind))
    return collector.locations


def summarise_kinds(locations: Sequence[Location]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for location in locations:
        counts[location.kind] = counts.get(location.kind, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
