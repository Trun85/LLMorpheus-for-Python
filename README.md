# LLMorpheus for Python

LLM-based mutation testing for Python — a from-scratch implementation of the
technique in **“LLMorpheus: Mutation Testing using Large Language Models”**
(Frank Tip, Jonathan Bell, Max Schäfer, [arXiv:2404.09952](https://arxiv.org/abs/2404.09952)).

Instead of applying a fixed set of mutation operators (`+` → `-`, delete a
statement), a placeholder is inserted at designated locations in the source and
an LLM is asked *what could go here instead*. The suggestions are filtered down
to syntactically valid, non-duplicate mutants, each is spliced back into the
file, and the test suite decides whether it is **killed**, **survived**, or
**timed out**.

The original tool targets JavaScript and delegates execution to a patched
StrykerJS. This project reimplements all three components — prompt generator,
mutant generator, and mutation runner — for Python, with **no third-party
dependencies**: only the standard library (`ast`, `urllib`, `subprocess`).

## Quick start

No API key needed — the bundled `mock` provider generates mutants offline so you
can watch the whole pipeline run:

```bash
cd examples/pricing && python -m llmorpheus all --provider mock --src sample_pkg
```

```
==> 1 source file(s), 42 placeholder location(s), 42 prompt(s)
==> 119 mutant(s) from 126 candidate(s) (discarded: 7 invalid, 0 identical, 0 duplicate)
==> running the test suite unmutated (baseline)
==> killed 95, survived 24, timed out 0 - mutation score 79.83%

file                   mutants   killed survived timeout   score
---------------------------------------------------------------
sample_pkg/pricing.py      119       95       24       0   79.8%
```

With a real model:

```bash
export OPENAI_API_KEY=sk-...
python -m llmorpheus all --src src --provider openai --model gpt-4o-mini --limit 50
```

Install it as a command (`llmorpheus`) with `pip install -e .`. Requires Python
3.9+, and a project with a green test suite — `pytest` by default, anything else
via `--test-command`.

## Commands

| Command | What it does |
| --- | --- |
| `locations` | List the placeholder locations that would be prompted about. Free. |
| `generate` | Prompt the LLM, filter suggestions, write `mutants.json`. |
| `run` | Apply each mutant, run the tests, write `results.json` + `report.html`. |
| `report` | Re-render a report from an existing `results.json`. |
| `all` | `generate` + `run` + `report`. |
| `templates` | List bundled templates and placeholder kinds. |

Always start with `locations` — it costs nothing and shows exactly what you
would be paying to mutate:

```bash
python -m llmorpheus locations --src src
```

## Choosing an LLM

`--provider openai` speaks the OpenAI chat-completions API, so `--base-url`
points it at anything compatible:

```bash
# OpenRouter (the paper used llama-3.3-70b-instruct and codellama variants)
python -m llmorpheus all --src src --provider openai \
  --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --model meta-llama/llama-3.3-70b-instruct

# A local model — no key required
python -m llmorpheus all --src src --provider openai \
  --base-url http://localhost:11434/v1 --model qwen2.5-coder

# Anthropic
python -m llmorpheus all --src src --provider anthropic --model claude-sonnet-4-5

# DeepSeek (defaults to deepseek-flash)
export DEEPSEEK_API_KEY=sk-...
python -m llmorpheus all --src src --provider deepseek
```

`deepseek-flash` runs in *thinking mode* by default on DeepSeek's side; the
`deepseek` provider turns it **off** unless you pass `--thinking`. In thinking
mode DeepSeek ignores `--temperature`, and the reasoning shares the output
budget with the answer — at the default 250 tokens it can use it all up and
return nothing — so `--thinking` also raises the default `--max-tokens` to 4096.
A one-line placeholder replacement rarely benefits from the extra reasoning,
and every reasoning token is billed.

Completions are cached under `.llmorpheus/cache/`, keyed by prompt, model,
temperature and token budget, so re-running after a crash costs nothing.
`--no-cache` disables it.

## Configuration: `.env` and subjects

API keys and your default LLM settings live in `.env`, which is gitignored:

```bash
cp .env.example .env
```

Each project under test gets a tracked `subjects/<name>.conf` recording where
its code is, what to exclude, and how to run its tests. A run is then just:

```bash
./.venv/bin/llmorpheus all --subject typesystem --limit 20
```

Both files use `KEY=VALUE` lines. Precedence, highest first: command-line
flags, the shell environment, the subject file, `.env`, built-in defaults.
`llmorpheus config --subject <name>` prints every effective value and where it
came from, with keys masked.

The loader enforces a few rules so a key cannot leak by accident:

- API keys are read only from `.env` or the environment. A subject file that
  contains one is rejected, since subject files are meant to be committed.
- Nothing is exported into `os.environ`, and `*_API_KEY` variables are removed
  from the environment of the test suite, so a mutated test that prints its
  environment cannot copy a key into `results.json`.
- A misspelt `LLMORPHEUS_*` key is an error that suggests the right name.

When a subject pins a version, `generate` and `run` warn if the checkout is at
a different tag or has modified tracked files, and record the commit in
`mutants.json`. [subjects/README.md](subjects/README.md) has the setup steps
for isort, mlxtend and typesystem.

## Useful flags

```bash
--kinds all                    # widen the placeholder scheme
--limit 50                     # cap the number of prompts (trial runs)
--temperature 0.5              # 0.0 is the default
--template template-no-hints   # prompt ablations
--jobs 8                       # evaluate mutants in parallel
--test-command "pytest -x -q tests/unit"
--test-timeout 60              # default: max(30s, 3 x baseline)
--fail-under 70                # non-zero exit if the score is too low (CI)
--rate-limit 500 --attempts 5  # for rate-limited endpoints
```

## Output

Each run lands in `.llmorpheus/run-<timestamp>/`, with `.llmorpheus/latest`
symlinked to the most recent one:

```
.llmorpheus/latest/
├── mutants.json      # mutants + generation stats (Table 2 of the paper)
├── results.json      # per-mutant killed/survived/timeout + test output
├── report.html       # self-contained, filterable
└── prompts/          # every prompt and completion, for auditing
```

`mutants.json` records a digest of every file it was generated from. If you edit
a file and then `run` an older mutant set, its mutants are skipped with a
warning rather than being applied at offsets that have since shifted.

## Layout

```
llmorpheus/
├── analysis/     parse sources, decide where placeholders go
├── prompting/    instantiate prompt templates (+ the paper's ablations)
├── llm/          OpenAI-compatible, Anthropic, DeepSeek, and an offline mock
├── generation/   completions → validated, deduplicated mutants
├── execution/    apply mutants, run tests, classify, report
├── settings.py   .env and subject files, layered
└── cli.py
subjects/         <name>.conf per project under test; clones go in <name>/
docs/             design notes and the placeholder reference
examples/pricing/ a tiny project with a deliberately incomplete test suite
tests/            the tool's own test suite
```

- **[docs/design.md](docs/design.md)** — how each component maps onto the paper,
  where the port had to diverge, how to reproduce the paper's experiments, and
  the known limitations.
- **[docs/placeholders.md](docs/placeholders.md)** — every placeholder kind,
  with examples.

## Tests

```bash
python -m pytest tests -q
```

Covers placeholder detection (including the byte-offset and f-string edge
cases), completion parsing and filtering, prompt construction, and a full
generate → run → report cycle against a temporary project.

## Credits

The technique is due to Tip, Bell and Schäfer; this is an independent
implementation for Python, not affiliated with the authors.
