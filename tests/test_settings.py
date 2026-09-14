"""Settings: .env and subject files, their precedence, and keeping secrets secret."""

import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from llmorpheus.cli import _client, _subject_status, command_config, parse_arguments
from llmorpheus.console import Console
from llmorpheus.execution.runner import RunnerConfig, TestRunner, resolve_test_command
from llmorpheus.llm.base import api_key_from_env
from llmorpheus.settings import SettingsError, load_settings, mask, parse_env_text

DOTENV_KEY = "sk-from-dotenv-0000abcd"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A stand-in repository root with a .env and one subject."""
    root = tmp_path / "repo"
    write(
        root / ".env",
        "DEEPSEEK_API_KEY={}\n"
        "LLMORPHEUS_PROVIDER=deepseek\n"
        "LLMORPHEUS_CONCURRENCY=2\n"
        "LLMORPHEUS_JOBS=3\n".format(DOTENV_KEY),
    )
    write(
        root / "subjects" / "demo.conf",
        "LLMORPHEUS_REPO=https://example.invalid/demo.git\n"
        "LLMORPHEUS_VERSION=1.2.3\n"
        "LLMORPHEUS_PROJECT=demo\n"
        "LLMORPHEUS_SRC=pkg\n"
        "LLMORPHEUS_EXCLUDE=pkg/vendored/**,pkg/old/**\n"
        "LLMORPHEUS_JOBS=1\n"
        "LLMORPHEUS_TEST_COMMAND=.venv/bin/python -m pytest -x -q\n"
        "LLMORPHEUS_TEST_ENV=CI=1 MPLBACKEND=Agg\n",
    )
    return root


# -- file format ------------------------------------------------------------
def test_env_file_format():
    text = (
        "# a comment\n"
        "\n"
        "export A=1\n"
        "B = spaced value   # trailing comment\n"
        "C='single # kept'\n"
        'D="line\\nbreak \\"quoted\\""\n'
        "E=>=3.10\n"
        "F=x#y\n"
        "G=\n"
        "H=CI=1 MPLBACKEND=Agg\n"
    )
    assert parse_env_text(text, "t.env") == {
        "A": "1",
        "B": "spaced value",
        "C": "single # kept",
        "D": 'line\nbreak "quoted"',
        "E": ">=3.10",
        "F": "x#y",
        "G": "",
        "H": "CI=1 MPLBACKEND=Agg",
    }


def test_malformed_line_names_the_position_but_never_echoes_it():
    with pytest.raises(SettingsError) as error:
        parse_env_text("OK=1\nDEEPSEEK_API_KEY sk-must-not-be-printed\n", "x.env")
    assert "x.env:2" in str(error.value)
    assert "sk-must-not-be-printed" not in str(error.value)


# -- layering ----------------------------------------------------------------
def test_precedence_environment_then_subject_then_dotenv(repo):
    settings = load_settings(
        subject="demo", environ={"LLMORPHEUS_CONCURRENCY": "9"}, repo_root=repo
    )
    assert settings.get("LLMORPHEUS_CONCURRENCY") == 9  # environment beats .env
    assert settings.get("LLMORPHEUS_JOBS") == 1  # subject beats .env
    assert settings.get("LLMORPHEUS_PROVIDER") == "deepseek"  # only in .env
    assert settings.get("LLMORPHEUS_MODEL") is None  # unset: the built-in default applies


def test_subject_values_are_typed_and_paths_resolve_against_the_subject_file(repo):
    settings = load_settings(subject="demo", environ={}, repo_root=repo)
    assert settings.subject_name == "demo"
    assert settings.get("LLMORPHEUS_PROJECT") == str(repo / "subjects" / "demo")
    assert settings.get("LLMORPHEUS_EXCLUDE") == ["pkg/vendored/**", "pkg/old/**"]
    assert settings.get("LLMORPHEUS_TEST_ENV") == {"CI": "1", "MPLBACKEND": "Agg"}


def test_unknown_subject_suggests_the_closest_one(repo):
    with pytest.raises(SettingsError) as error:
        load_settings(subject="dem", environ={}, repo_root=repo)
    assert "did you mean 'demo'" in str(error.value)


def test_explicit_env_file_must_exist(repo, tmp_path):
    with pytest.raises(SettingsError):
        load_settings(env_file=str(tmp_path / "missing.env"), environ={}, repo_root=repo)


def test_empty_values_count_as_unset(repo):
    write(repo / ".env", "DEEPSEEK_API_KEY=\nLLMORPHEUS_MODEL=\n")
    settings = load_settings(environ={}, repo_root=repo)
    assert settings.get("LLMORPHEUS_MODEL") is None
    assert settings.secrets() == {}


# -- validation ----------------------------------------------------------------
def test_subject_files_refuse_secrets(repo):
    write(
        repo / "subjects" / "leaky.conf",
        "LLMORPHEUS_SRC=pkg\nDEEPSEEK_API_KEY=sk-oops-oops-oops\n",
    )
    with pytest.raises(SettingsError) as error:
        load_settings(subject="leaky", environ={}, repo_root=repo)
    assert "put it in .env" in str(error.value)
    assert "sk-oops" not in str(error.value)


def test_misspelt_setting_is_rejected_with_a_suggestion(repo):
    write(repo / ".env", "LLMORPHEUS_JOB=2\n")
    with pytest.raises(SettingsError) as error:
        load_settings(environ={}, repo_root=repo)
    assert "did you mean LLMORPHEUS_JOBS?" in str(error.value)


@pytest.mark.parametrize(
    "line", ["LLMORPHEUS_JOBS=two", "LLMORPHEUS_THINKING=maybe", "LLMORPHEUS_PROVIDER=gemini"]
)
def test_invalid_values_fail_as_soon_as_the_file_is_loaded(repo, line):
    write(repo / ".env", line + "\n")
    with pytest.raises(SettingsError) as error:
        load_settings(environ={}, repo_root=repo)
    assert line.split("=")[0] in str(error.value)


# -- command line ---------------------------------------------------------------
def test_cli_takes_defaults_from_the_subject_and_flags_still_win(repo):
    args = parse_arguments(["all", "--subject", "demo"], environ={}, repo_root=repo)
    assert args.project == str(repo / "subjects" / "demo")
    assert args.src == ["pkg"]
    assert args.exclude == ["pkg/vendored/**", "pkg/old/**"]
    assert (args.jobs, args.concurrency, args.provider) == (1, 2, "deepseek")
    assert args.test_env == {"CI": "1", "MPLBACKEND": "Agg"}

    args = parse_arguments(
        [
            "all", "--subject", "demo", "--jobs", "4", "--src", "pkg/core",
            "--test-env", "CI=0", "--provider", "mock",
        ],
        environ={},
        repo_root=repo,
    )
    # --src replaces the subject's sources rather than appending to them.
    assert (args.jobs, args.src, args.provider) == (4, ["pkg/core"], "mock")
    assert args.test_env == {"CI": "0", "MPLBACKEND": "Agg"}


def test_subject_works_before_the_subcommand_too(repo):
    args = parse_arguments(["--subject", "demo", "locations"], environ={}, repo_root=repo)
    assert args.src == ["pkg"]


def test_no_thinking_overrides_a_dotenv_that_turns_it_on(repo):
    write(repo / ".env", "LLMORPHEUS_THINKING=true\n")
    assert parse_arguments(["generate"], environ={}, repo_root=repo).thinking is True
    assert (
        parse_arguments(["generate", "--no-thinking"], environ={}, repo_root=repo).thinking
        is False
    )


def test_config_command_shows_sources_and_masks_keys(repo):
    stream = io.StringIO()
    args = parse_arguments(["config", "--subject", "demo"], environ={}, repo_root=repo)
    assert command_config(args, Console(stream=stream, use_colour=False)) == 0
    output = stream.getvalue()
    assert DOTENV_KEY not in output
    assert "set (…abcd)" in output
    assert "demo.conf" in output
    assert "LLMORPHEUS_JOBS" in output


# -- secrets ------------------------------------------------------------------------
def test_api_key_from_dotenv_is_used_but_never_exposed(repo, tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    args = parse_arguments(["generate", "--no-cache"], environ={}, repo_root=repo)
    client = _client(args, tmp_path)

    client.preflight()  # finds the key in .env, so does not raise
    assert DOTENV_KEY not in repr(client)
    assert DOTENV_KEY not in json.dumps(client.describe())  # describe() feeds mutants.json

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-the-shell-1234")
    assert (
        api_key_from_env("DEEPSEEK_API_KEY", "deepseek", client.secrets)
        == "sk-from-the-shell-1234"
    )


def test_llm_credentials_never_reach_the_test_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shell-value-0001")
    monkeypatch.setenv("SOME_OTHER_API_KEY", "x")
    monkeypatch.setenv("CUSTOM_KEY_VARIABLE", "y")
    monkeypatch.setenv("UNRELATED_SETTING", "keep")
    runner = TestRunner(
        RunnerConfig(project_root=tmp_path, env={"CI": "1"}, scrub_env=("CUSTOM_KEY_VARIABLE",))
    )
    env = runner._environment()
    assert not {"DEEPSEEK_API_KEY", "SOME_OTHER_API_KEY", "CUSTOM_KEY_VARIABLE"} & set(env)
    assert env["UNRELATED_SETTING"] == "keep"
    assert env["CI"] == "1"


def test_mask_reveals_at_most_the_last_four_characters():
    assert mask("") == "not set"
    assert mask("sk-0123456789abcdef") == "set (…cdef)"
    assert mask("short") == "set"


# -- test command -----------------------------------------------------------------
def test_relative_interpreter_is_anchored_at_the_project_without_following_symlinks(tmp_path):
    project = tmp_path / "project with spaces"
    base = write(tmp_path / "base-python", "")
    (project / ".venv" / "bin").mkdir(parents=True)
    os.symlink(str(base), str(project / ".venv" / "bin" / "python"))

    command, use_shell = resolve_test_command(".venv/bin/python -m pytest -x -q", project)

    assert use_shell is False
    # The venv's own path, not the base interpreter it links to.
    assert command == [str(project / ".venv" / "bin" / "python"), "-m", "pytest", "-x", "-q"]


def test_project_placeholder_is_shell_quoted(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    command, use_shell = resolve_test_command("cd {project} && python -m pytest -q", project)
    assert use_shell is True
    assert "cd '{}' &&".format(project) in command[0]


def test_missing_interpreter_is_reported_before_anything_runs(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        resolve_test_command(".venv/bin/python -m pytest", tmp_path)
    assert "virtual environment" in str(error.value)


# -- provenance -------------------------------------------------------------------
def test_subject_status_checks_the_pinned_version_and_local_edits(repo):
    project = repo / "subjects" / "demo"
    write(project / "pkg" / "module.py", "X = 1\n")

    def git(*arguments):
        subprocess.run(
            [
                "git", "-C", str(project),
                "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
            ]
            + list(arguments),
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    git("tag", "1.2.3")

    args = parse_arguments(["run", "--subject", "demo"], environ={}, repo_root=repo)
    stream = io.StringIO()
    console = Console(stream=stream, use_colour=False)

    info = _subject_status(args, project, console)
    assert info["tag"] == "1.2.3" and info["modified"] is False
    assert stream.getvalue() == ""

    write(project / "pkg" / "module.py", "X = 2\n")
    info = _subject_status(args, project, console)
    assert info["modified"] is True
    assert "local modifications" in stream.getvalue()
