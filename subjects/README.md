# Subjects

Each project under test has two parts here:

- `<name>.conf` - tracked. Where the code is, what to exclude, how to run the
  tests, and the pinned version.
- `<name>/` - a clone of the project at that version, with its own `.venv`.
  Gitignored.

llmorpheus itself runs from this repository's `.venv`. Each subject's tests run
with that subject's own interpreter, named in `LLMORPHEUS_TEST_COMMAND`, so the
two Pythons never have to match. Python 3.9 parses all three code bases below; a
subject written with 3.10+ syntax such as `match` would need llmorpheus itself to
run on a newer Python.

| Subject | Version | Needs Python | Status on this machine |
|---|---|---|---|
| typesystem | 0.4.1 | 3.6 - 3.10 | verified: all 1121 tests pass with the system Python 3.9 |
| isort | 7.0.0 | >= 3.10 | not yet verified |
| mlxtend | v0.25.0 | >= 3.11 | not yet verified |

Every command below runs from the repository root.

## 1. API key

Put your key after `DEEPSEEK_API_KEY=` in `.env`. That file is gitignored and
readable only by you; never put a key anywhere else.

## 2. Python 3.12 (isort and mlxtend only)

Install it from <https://www.python.org/downloads/macos/>. The installer adds
`python3.12` to `/usr/local/bin`:

```bash
python3.12 --version
```

typesystem deliberately uses the system Python at `/usr/bin/python3` instead:
upstream last tested it on Python 3.10 and earlier.

## 3. typesystem

```bash
git clone --branch 0.4.1 --depth 1 https://github.com/encode/typesystem.git subjects/typesystem
```

```bash
/usr/bin/python3 -m venv subjects/typesystem/.venv
```

```bash
subjects/typesystem/.venv/bin/python -m pip install --upgrade pip
```

```bash
subjects/typesystem/.venv/bin/python -m pip install -e subjects/typesystem pytest==8.3.5 jinja2==3.1.6 pyyaml==6.0.3
```

Why `pytest==8.3.5`: typesystem's `setup.cfg` turns every warning into an error,
and pytest 8.4 reports the suite's unclosed JSON files as warnings - so with the
newest pytest the unmodified suite fails before any mutant runs. jinja2 and
pyyaml are pinned to the versions verified to pass.

## 4. isort

```bash
git clone --branch 7.0.0 --depth 1 https://github.com/PyCQA/isort.git subjects/isort
```

```bash
python3.12 -m venv subjects/isort/.venv
```

```bash
subjects/isort/.venv/bin/python -m pip install --upgrade pip
```

```bash
subjects/isort/.venv/bin/python -m pip install -e subjects/isort --group subjects/isort/pyproject.toml:dev
```

`--group` needs pip 25.1 or newer, hence the upgrade. `dev` is isort's own
development group - the test dependencies plus linters, docs tooling and tox. It
is large, but it is what isort's CI installs.

## 5. mlxtend

```bash
git clone --branch v0.25.0 --depth 1 https://github.com/rasbt/mlxtend.git subjects/mlxtend
```

```bash
python3.12 -m venv subjects/mlxtend/.venv
```

```bash
subjects/mlxtend/.venv/bin/python -m pip install --upgrade pip
```

```bash
subjects/mlxtend/.venv/bin/python -m pip install -e subjects/mlxtend pytest
```

## 6. Check, then run

```bash
./.venv/bin/llmorpheus config --subject typesystem
```

```bash
./.venv/bin/llmorpheus locations --subject typesystem
```

```bash
./.venv/bin/llmorpheus all --subject typesystem --limit 20
```

`all` and `run` start by running the unmodified suite, and stop if it fails.
Every mutant repeats that run, so time a slow subject's suite yourself before a
large run:

```bash
cd subjects/mlxtend && time env MPLBACKEND=Agg .venv/bin/python -m pytest mlxtend -x -q
```

## Adding a subject

Copy an existing `.conf` and set:

| Key | Meaning |
|---|---|
| `LLMORPHEUS_REPO`, `LLMORPHEUS_VERSION` | Where the project comes from and the tag to check out. `generate` and `run` warn when the clone is at another tag or has modified tracked files. |
| `LLMORPHEUS_PROJECT` | The clone, relative to the `.conf` file. |
| `LLMORPHEUS_SRC` | Package directories to mutate, comma-separated, relative to the project. |
| `LLMORPHEUS_EXCLUDE` | Globs never to mutate: bundled third-party code, and code whose tests the project never runs. Test files are always excluded. |
| `LLMORPHEUS_TEST_COMMAND` | The project's own test command, plus `-x`. A relative `.venv/bin/python` is resolved against the project. |
| `LLMORPHEUS_TEST_ENV` | `KEY=VALUE` pairs for the test run, such as `CI=1` for derandomized Hypothesis. |
| `LLMORPHEUS_JOBS`, `LLMORPHEUS_TEST_TIMEOUT` | Parallel workers, and the per-mutant timeout (default: 3 times the baseline, at least 30 s). |

The other `LLMORPHEUS_*` settings in `.env.example` can be overridden per subject
as well. API keys cannot: a subject file that contains one is rejected.
