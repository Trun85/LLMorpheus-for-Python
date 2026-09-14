"""Finding the source files of a project that should be mutated."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    ".llmorpheus",
)

DEFAULT_EXCLUDE_GLOBS = (
    "test_*.py",
    "*_test.py",
    "tests/**",
    "**/tests/**",
    "test/**",
    "**/test/**",
    "conftest.py",
    "**/conftest.py",
    "setup.py",
    "**/_version.py",
    "**/version.py",
    "**/__main__.py",
)


@dataclass
class SourceFile:
    """A file to be mutated, addressed by its project-relative path."""

    relative_path: str
    absolute_path: Path
    source: str

    @property
    def line_count(self) -> int:
        return self.source.count("\n") + 1


@dataclass
class FileSelector:
    """Selects which ``.py`` files under a project root are mutation targets."""

    project_root: Path
    sources: Sequence[str] = field(default_factory=tuple)
    include: Sequence[str] = field(default_factory=tuple)
    exclude: Sequence[str] = field(default_factory=tuple)
    use_default_excludes: bool = True

    def _exclude_globs(self) -> List[str]:
        globs = list(self.exclude)
        if self.use_default_excludes:
            globs.extend(DEFAULT_EXCLUDE_GLOBS)
        return globs

    def _is_excluded(self, relative: str) -> bool:
        name = os.path.basename(relative)
        for pattern in self._exclude_globs():
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern):
                return True
            # ``tests/**`` should also match the directory itself.
            if pattern.endswith("/**") and (
                relative == pattern[:-3] or relative.startswith(pattern[:-2])
            ):
                return True
        return False

    def _is_included(self, relative: str) -> bool:
        if not self.include:
            return True
        name = os.path.basename(relative)
        return any(
            fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(name, pattern)
            for pattern in self.include
        )

    def _roots(self) -> List[Path]:
        if not self.sources:
            return [self.project_root]
        roots = []
        for entry in self.sources:
            path = Path(entry)
            roots.append(path if path.is_absolute() else self.project_root / path)
        return roots

    def find(self) -> List[SourceFile]:
        seen = set()
        found: List[SourceFile] = []
        for root in self._roots():
            if not root.exists():
                raise FileNotFoundError("source path does not exist: {}".format(root))
            for path in _walk_python_files(root):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                try:
                    relative = resolved.relative_to(self.project_root.resolve()).as_posix()
                except ValueError:
                    # An absolute path here would defeat the per-worker project
                    # copies used by --jobs: ``worker_root / "/abs/path"`` is
                    # just "/abs/path", so every worker would mutate the one
                    # real file at once. Every mutation target must live under
                    # the project root.
                    raise ValueError(
                        "source path {} is outside the project root {}; "
                        "point --project at a directory that contains it".format(
                            resolved, self.project_root
                        )
                    )
                if not self._is_included(relative) or self._is_excluded(relative):
                    continue
                try:
                    source = resolved.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                seen.add(resolved)
                found.append(
                    SourceFile(relative_path=relative, absolute_path=resolved, source=source)
                )
        found.sort(key=lambda item: item.relative_path)
        return found


def _walk_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in DEFAULT_EXCLUDE_DIRS and not name.startswith(".")
        ]
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def guess_source_dirs(project_root: Path) -> List[str]:
    """Best-effort guess of where a project keeps its code.

    Used when the user does not pass ``--src``; prefers ``src/`` and top-level
    packages over sweeping the whole tree.
    """
    candidates: List[str] = []
    src = project_root / "src"
    if src.is_dir():
        candidates.append("src")
    for entry in sorted(project_root.iterdir()) if project_root.is_dir() else []:
        if entry.name in DEFAULT_EXCLUDE_DIRS or entry.name.startswith("."):
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            candidates.append(entry.name)
    return candidates or ["."]


def resolve_project_root(value: Optional[str]) -> Path:
    root = Path(value or ".").expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError("project root is not a directory: {}".format(root))
    return root
