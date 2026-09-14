"""Layered configuration: ``.env`` for secrets and defaults, ``subjects/*.conf`` per project.

Precedence, highest first:

1. command-line flags
2. the process environment (``LLMORPHEUS_JOBS=4 llmorpheus ...``)
3. the subject file chosen with ``--subject`` (``subjects/<name>.conf``)
4. the ``.env`` file
5. built-in defaults

Both files use the same ``KEY=VALUE`` format, parsed here with the standard
library: Python 3.9 has no ``tomllib``, and the tool has no dependencies.

Values are read, never exported. Nothing from a file is placed in
``os.environ``, so API keys cannot reach the test suites that llmorpheus runs
against mutated third-party code - where a failing test that prints its
environment would otherwise copy them into ``results.json``.
"""

from __future__ import annotations

import difflib
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBJECT_SUFFIX = ".conf"
SUBJECTS_DIR_KEY = "LLMORPHEUS_SUBJECTS_DIR"
ENV_FILE_KEY = "LLMORPHEUS_ENV_FILE"

PROVIDER_KEY_NAMES = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class SettingsError(ValueError):
    """A settings file, or a value in one, is malformed."""


@dataclass(frozen=True)
class Option:
    key: str
    dest: Optional[str]  # argparse destination; None for informational keys
    kind: str  # str, int, float, bool, list, path, env, provider


OPTIONS: Tuple[Option, ...] = (
    # project and mutation targets
    Option("LLMORPHEUS_PROJECT", "project", "path"),
    Option("LLMORPHEUS_SRC", "src", "list"),
    Option("LLMORPHEUS_INCLUDE", "include", "list"),
    Option("LLMORPHEUS_EXCLUDE", "exclude", "list"),
    Option("LLMORPHEUS_KINDS", "kinds", "list"),
    Option("LLMORPHEUS_TEMPLATE", "template", "str"),
    Option("LLMORPHEUS_SYSTEM_TEMPLATE", "system_template", "str"),
    # LLM
    Option("LLMORPHEUS_PROVIDER", "provider", "provider"),
    Option("LLMORPHEUS_MODEL", "model", "str"),
    Option("LLMORPHEUS_BASE_URL", "base_url", "str"),
    Option("LLMORPHEUS_API_KEY_ENV", "api_key_env", "str"),
    Option("LLMORPHEUS_TEMPERATURE", "temperature", "float"),
    Option("LLMORPHEUS_MAX_TOKENS", "max_tokens", "int"),
    Option("LLMORPHEUS_THINKING", "thinking", "bool"),
    Option("LLMORPHEUS_ATTEMPTS", "attempts", "int"),
    Option("LLMORPHEUS_RATE_LIMIT_MS", "rate_limit", "int"),
    Option("LLMORPHEUS_REQUEST_TIMEOUT", "request_timeout", "float"),
    Option("LLMORPHEUS_CONCURRENCY", "concurrency", "int"),
    # test execution
    Option("LLMORPHEUS_TEST_COMMAND", "test_command", "str"),
    Option("LLMORPHEUS_TEST_ENV", "test_env", "env"),
    Option("LLMORPHEUS_TEST_TIMEOUT", "test_timeout", "float"),
    Option("LLMORPHEUS_JOBS", "jobs", "int"),
    # provenance, recorded in mutants.json and checked against the checkout
    Option("LLMORPHEUS_REPO", None, "str"),
    Option("LLMORPHEUS_VERSION", None, "str"),
)
BY_KEY: Dict[str, Option] = {option.key: option for option in OPTIONS}
BY_DEST: Dict[str, Option] = {option.dest: option for option in OPTIONS if option.dest}
_NON_SCALAR = ("list", "env")


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return upper in PROVIDER_KEY_NAMES or upper.endswith(_SECRET_SUFFIXES)


def mask(value: Optional[str]) -> str:
    """Describe a secret without revealing it."""
    text = (value or "").strip()
    if not text:
        return "not set"
    return "set (…{})".format(text[-4:]) if len(text) >= 12 else "set"


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_env_text(text: str, source: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines.

    Supports blank lines, ``#`` comments, an optional ``export`` prefix,
    single- or double-quoted values (double quotes understand ``\\n``, ``\\t``,
    ``\\"`` and ``\\\\``), and ``#`` comments after unquoted values when preceded
    by whitespace. No variable interpolation and no multi-line values.

    Errors name the file and line but never repeat the line itself: a malformed
    line may well be an API key with a typo in it.
    """
    values: Dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, rest = line.partition("=")
        key = key.strip()
        where = "{}:{}".format(source, number)
        if not separator or not _KEY.match(key):
            raise SettingsError("{}: expected KEY=VALUE".format(where))
        values[key] = _parse_value(rest.strip(), where)
    return values


def _parse_value(text: str, where: str) -> str:
    if not text:
        return ""
    quote = text[0]
    if quote not in ("'", '"'):
        comment = re.search(r"\s#", text)
        return (text[: comment.start()] if comment else text).strip()
    escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    out: List[str] = []
    index = 1
    while index < len(text):
        char = text[index]
        if quote == '"' and char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            out.append(escapes.get(following, "\\" + following))
            index += 2
            continue
        if char == quote:
            trailing = text[index + 1 :].strip()
            if trailing and not trailing.startswith("#"):
                raise SettingsError("{}: unexpected text after the closing quote".format(where))
            return "".join(out)
        out.append(char)
        index += 1
    raise SettingsError("{}: unterminated quoted value".format(where))


def parse_env_file(path: Path) -> Dict[str, str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SettingsError("{}: not valid UTF-8".format(path))
    except OSError as error:
        raise SettingsError("cannot read {}: {}".format(path, error))
    return parse_env_text(text, str(path))


def parse_env_assignments(text: str, where: str) -> Dict[str, str]:
    """Parse ``KEY=VALUE KEY2="value two"`` into a dict (shell quoting rules)."""
    try:
        tokens = shlex.split(text)
    except ValueError as error:
        raise SettingsError("{}: {}".format(where, error))
    result: Dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator or not _KEY.match(key):
            raise SettingsError("{}: expected KEY=VALUE pairs separated by spaces".format(where))
        result[key] = value
    return result


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
@dataclass
class Layer:
    name: str  # "environment", "subject" or ".env"
    path: Optional[Path]
    values: Dict[str, str]
    base_dir: Path  # relative paths in this layer resolve against it

    @property
    def label(self) -> str:
        return self.name if self.path is None else "{} {}".format(self.name, self.path)


def _convert(option: Option, raw: str, layer: Layer) -> object:
    where = "{} in {}".format(option.key, layer.label)
    value = raw.strip()
    if option.kind in ("int", "float"):
        try:
            return int(value) if option.kind == "int" else float(value)
        except ValueError:
            raise SettingsError("{}: expected a number, got {!r}".format(where, value))
    if option.kind == "bool":
        lowered = value.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise SettingsError("{}: expected true or false, got {!r}".format(where, value))
    if option.kind == "list":
        return [part.strip() for part in value.split(",") if part.strip()]
    if option.kind == "path":
        path = Path(value).expanduser()
        # abspath, not resolve(): never follow symlinks behind the user's back.
        return str(path) if path.is_absolute() else os.path.abspath(str(layer.base_dir / path))
    if option.kind == "env":
        return parse_env_assignments(value, where)
    if option.kind == "provider":
        from .llm import PROVIDERS

        if value.lower() not in PROVIDERS:
            raise SettingsError(
                "{}: unknown provider {!r}; choose one of: {}".format(
                    where, value, ", ".join(PROVIDERS)
                )
            )
        return value.lower()
    return value


def _validate(layer: Layer) -> None:
    """Reject typos, misplaced secrets and bad values when a file is loaded."""
    for key, value in layer.values.items():
        if key.startswith("LLMORPHEUS_"):
            if key == ENV_FILE_KEY:
                raise SettingsError(
                    "{}: {} only works as an environment variable".format(layer.label, key)
                )
            if key == SUBJECTS_DIR_KEY:
                if layer.name == "subject":
                    raise SettingsError(
                        "{}: {} has no effect in a subject file".format(layer.label, key)
                    )
                continue
            if key not in BY_KEY:
                close = difflib.get_close_matches(key, sorted(BY_KEY) + [SUBJECTS_DIR_KEY], n=1)
                raise SettingsError(
                    "{}: unknown setting {}{}".format(
                        layer.label, key, "; did you mean {}?".format(close[0]) if close else ""
                    )
                )
            if value.strip():
                _convert(BY_KEY[key], value, layer)
        elif layer.name == "subject":
            if is_secret_name(key):
                raise SettingsError(
                    "{}: {} looks like a secret. Subject files are meant to be committed, so "
                    "secrets are never read from them - put it in .env instead.".format(
                        layer.label, key
                    )
                )
            raise SettingsError(
                "{}: only LLMORPHEUS_* settings belong in a subject file, found {}".format(
                    layer.label, key
                )
            )


def _display(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(value)
    if isinstance(value, dict):
        return " ".join("{}={}".format(key, item) for key, item in value.items())
    return str(value)


@dataclass
class Settings:
    """Settings layers, highest precedence first (command-line flags excluded)."""

    layers: List[Layer]
    env_file: Optional[Path] = None
    subject_name: Optional[str] = None
    subject_file: Optional[Path] = None
    subjects_dir: Path = field(default_factory=lambda: REPO_ROOT / "subjects")

    def source_of(self, key: str) -> Optional[Layer]:
        for layer in self.layers:
            value = layer.values.get(key)
            if value is not None and value.strip():
                return layer
        return None

    def get(self, key: str) -> object:
        """The typed value of a known option, or ``None`` when nothing sets it."""
        layer = self.source_of(key)
        if layer is None:
            return None
        return _convert(BY_KEY[key], layer.values[key], layer)

    def scalar_defaults(self) -> Dict[str, object]:
        """argparse ``set_defaults`` values for every single-valued option that is set."""
        defaults: Dict[str, object] = {}
        for option in OPTIONS:
            if option.dest is None or option.kind in _NON_SCALAR:
                continue
            value = self.get(option.key)
            if value is not None:
                defaults[option.dest] = value
        return defaults

    def list_default(self, dest: str) -> object:
        option = BY_DEST.get(dest)
        return None if option is None else self.get(option.key)

    def secrets(self, extra_names: Sequence[str] = ()) -> Dict[str, str]:
        """Secret values from the files. The process environment is read separately, and wins."""
        wanted = set(extra_names)
        found: Dict[str, str] = {}
        for layer in reversed(self.layers):  # lowest precedence first; higher layers overwrite
            if layer.name == "environment":
                continue
            for key, value in layer.values.items():
                if (is_secret_name(key) or key in wanted) and value.strip():
                    found[key] = value.strip()
        return found

    def describe_layer(self, layer: Layer) -> str:
        if layer.name == "environment":
            return "shell environment"
        if layer.name == "subject":
            return "subject file   {}".format(layer.path)
        if layer.path is None:
            return ".env file      (none found; copy .env.example to {})".format(
                REPO_ROOT / ".env"
            )
        return ".env file      {}".format(layer.path)

    def rows(self, extra_secret_names: Sequence[str] = ()) -> List[Tuple[str, str, str]]:
        """``(key, display value, source)`` for every option and credential."""
        rows: List[Tuple[str, str, str]] = []
        for option in OPTIONS:
            layer = self.source_of(option.key)
            if layer is None:
                rows.append((option.key, "-", "default"))
            else:
                rows.append((option.key, _display(self.get(option.key)), layer.name))
        custom = self.get("LLMORPHEUS_API_KEY_ENV")
        names = list(PROVIDER_KEY_NAMES)
        for name in [custom] + list(extra_secret_names):
            if name and name not in names:
                names.append(str(name))
        for name in names:
            layer = self.source_of(name)
            rows.append(
                (name, mask(layer.values[name]) if layer else "not set", layer.name if layer else "-")
            )
        return rows


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def list_subjects(subjects_dir: Path) -> List[str]:
    if not subjects_dir.is_dir():
        return []
    return sorted(path.stem for path in subjects_dir.glob("*" + SUBJECT_SUFFIX))


def resolve_subject(value: str, subjects_dir: Path) -> Path:
    """``isort`` -> ``subjects/isort.conf``; a path to a ``.conf`` file is used as is."""
    text = value.strip()
    if text.endswith(SUBJECT_SUFFIX) or "/" in text or os.sep in text:
        path = Path(text).expanduser()
        path = path if path.is_absolute() else Path.cwd() / path
        if not path.is_file():
            raise SettingsError("subject file not found: {}".format(path))
        return path
    path = subjects_dir / (text + SUBJECT_SUFFIX)
    if path.is_file():
        return path
    available = list_subjects(subjects_dir)
    close = difflib.get_close_matches(text, available, n=1)
    raise SettingsError(
        "unknown subject {!r}: {} does not exist{}; available: {}".format(
            text,
            path,
            " (did you mean {!r}?)".format(close[0]) if close else "",
            ", ".join(available) if available else "none",
        )
    )


def _find_env_file(
    explicit: Optional[str], environ: Mapping[str, str], root: Path, cwd: Path
) -> Optional[Path]:
    for candidate, origin in ((explicit, "--env-file"), (environ.get(ENV_FILE_KEY), ENV_FILE_KEY)):
        if candidate and candidate.strip():
            path = Path(candidate.strip()).expanduser()
            path = path if path.is_absolute() else cwd / path
            if not path.is_file():
                raise SettingsError("{} points to a file that does not exist: {}".format(origin, path))
            return path
    default = root / ".env"
    return default if default.is_file() else None


def _subjects_dir(environment: Layer, dotenv: Layer, root: Path) -> Path:
    for layer in (environment, dotenv):
        value = layer.values.get(SUBJECTS_DIR_KEY, "").strip()
        if value:
            path = Path(value).expanduser()
            return path if path.is_absolute() else Path(os.path.abspath(str(layer.base_dir / path)))
    return root / "subjects"


def load_settings(
    env_file: Optional[str] = None,
    subject: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> Settings:
    """Read the environment, the ``.env`` file and (optionally) a subject file."""
    environ = os.environ if environ is None else environ
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    cwd = Path.cwd()

    environment = Layer(
        "environment",
        None,
        {
            key: value
            for key, value in environ.items()
            if key in BY_KEY or key == SUBJECTS_DIR_KEY or is_secret_name(key)
        },
        cwd,
    )
    _validate(environment)

    dotenv_path = _find_env_file(env_file, environ, root, cwd)
    dotenv = Layer(
        ".env",
        dotenv_path,
        parse_env_file(dotenv_path) if dotenv_path else {},
        dotenv_path.parent if dotenv_path else root,
    )
    _validate(dotenv)

    subjects_dir = _subjects_dir(environment, dotenv, root)
    layers = [environment]
    subject_path: Optional[Path] = None
    if subject:
        subject_path = resolve_subject(subject, subjects_dir)
        subject_layer = Layer("subject", subject_path, parse_env_file(subject_path), subject_path.parent)
        _validate(subject_layer)
        layers.append(subject_layer)
    layers.append(dotenv)

    settings = Settings(
        layers=layers,
        env_file=dotenv_path,
        subject_name=subject_path.stem if subject_path else None,
        subject_file=subject_path,
        subjects_dir=subjects_dir,
    )
    # A custom --api-key-env name is not secret-looking, so pick it up explicitly.
    custom = settings.get("LLMORPHEUS_API_KEY_ENV")
    if isinstance(custom, str) and custom in environ:
        environment.values[custom] = environ[custom]
    return settings
