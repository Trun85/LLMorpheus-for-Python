# Design: porting LLMorpheus to Python

How the technique from [arXiv:2404.09952](https://arxiv.org/abs/2404.09952) is
realised here, and where the Python version had to diverge from the JavaScript
original.

## The pipeline

The paper's Figure 5 has three components working in concert. The package is
laid out along the same seams:

```
llmorpheus/
├── analysis/          parse sources, decide where placeholders go
│   ├── positions.py     AST positions → character offsets in the source
│   ├── locations.py     the placeholder scheme (see placeholders.md)
│   └── project.py       which files are mutation targets
├── prompting/         instantiate prompt templates around a placeholder
│   ├── prompts.py
│   └── templates/       the paper's template and its ablations
├── llm/               model back-ends
│   ├── base.py          retries, rate limiting, on-disk completion cache
│   ├── openai_client.py OpenAI-compatible: OpenAI, OpenRouter, vLLM, Ollama
│   ├── anthropic_client.py
│   ├── deepseek_client.py OpenAI-compatible, with thinking mode off by default
│   └── mock.py          deterministic offline stand-in
├── generation/        completions → mutants
│   ├── mutants.py       parsing, filtering, the mutants.json format
│   └── generator.py     orchestration: locations → prompts → mutants
├── execution/         the role of the patched StrykerJS
│   ├── runner.py        apply a mutant, run the tests, classify
│   └── report.py        terminal table + self-contained HTML report
├── console.py         progress output
└── cli.py             the command-line interface
```

Component by component:

| Paper (JavaScript) | Here (Python) |
| --- | --- |
| BabelJS parses sources and locates placeholders | `analysis` — stdlib `ast` plus a byte-accurate position mapper |
| Handlebars instantiates the prompt template | `prompting` — verbatim `{{code}}` / `{{orig}}` substitution |
| BabelJS re-parses each candidate for validity | `generation` — candidate is spliced into the whole file, which is re-parsed |
| Patched StrykerJS applies mutants and classifies them | `execution.runner` — applies one mutant at a time, runs your test command |
| StrykerJS's interactive HTML report | `execution.report` — one self-contained HTML file |

## Where the port diverges

### Validity can only be judged in context

StrykerJS parses a candidate fragment on its own. That does not work for
Python: `x = 1` parses fine in isolation but is invalid inside `if …:`, and
indentation makes standalone fragment parsing unreliable in general.

Every candidate is therefore spliced into the whole file and the file is
re-parsed with `ast.parse`. This is stricter than the original and also catches
candidates that would corrupt a loop header or an argument list. It is the
single most important adaptation in the port.

### Column offsets are bytes, not characters

`ast` reports `col_offset` as a byte offset into the UTF-8 encoding of the
line. Slicing a `str` with it directly corrupts any file containing non-ASCII
text — a real hazard for source with accented identifiers, comments or string
literals. `analysis.positions.SourceMap` is the one place that knows about
this, and converts properly.

### f-strings are left alone

Before Python 3.12 the nodes inside an f-string carry positions that do not
correspond to the real source text, so the traversal treats a `JoinedStr` as a
leaf. An f-string can still be replaced as a whole when it is itself an
argument or a condition.

### No single-AST-node constraint

StrykerJS requires each mutant to correspond to exactly one AST node, so the
paper's tool expands loop headers and argument lists to the nearest enclosing
node. Here mutants are plain text spans, so `f(<PLACEHOLDER>)` over a full
argument list needs no expansion.

### Single-line replacements by default

The paper's template asks for "a single line of code". Multi-line replacements
are rejected unless `--allow-multiline` is passed, which mostly increases the
number of invalid candidates.

## Execution model

- **Baseline first.** The unmutated suite must pass, otherwise every mutant
  would look killed. A red baseline aborts the run.
- **Stale mutants are refused.** A mutant records absolute line/column offsets,
  which only mean anything against the exact text they were generated from. If
  a file is edited between `generate` and `run`, applying the mutant would
  splice the replacement into an unrelated region and produce a confident but
  fabricated verdict. `mutants.json` therefore stores a digest per file;
  `run` skips mutants for changed files (reporting them as errored, excluded
  from the score) and `Workspace.apply` re-checks each span against the
  recorded original before writing.
- **In-place mutation.** A mutant is written into the real file and restored in
  a `finally` block, with an `atexit` hook and `SIGINT`/`SIGTERM` handlers as
  backstops. An interrupted run does not leave mutated source behind — but
  commit your work before a long run anyway.
- **Parallelism.** `--jobs N` above 1 copies the project into one temporary
  directory per worker and mutates the copies. If your package is installed in
  editable mode, the copy's root is prepended to `PYTHONPATH` so the mutated
  code wins; verify with a small `--limit` run before trusting a large one.
- **No stale bytecode.** `PYTHONDONTWRITEBYTECODE=1` is set for test
  subprocesses and any `.pyc` for a mutated module is deleted, so a cached
  compile can never mask a mutant.
- **Timeout.** Defaults to `max(30s, 3 × baseline)`. Timed-out mutants count as
  detected, as in Stryker.
- **Mutation score.** `(killed + timeout) / (killed + timeout + survived)`,
  matching the Stryker definition the paper uses. Mutants that could not be
  evaluated are reported separately and excluded from the denominator.

## Reproducing the paper's experiments

- **RQ3 (temperature).** `--temperature 0.0 | 0.25 | 0.5 | 1.0`. Pass
  `--no-cache` when repeating a run at the same temperature, or the cached
  completion is reused.
- **RQ4 (prompt ablations).** `--template` selects among `template-full`
  (everything, the default), `template-no-explanation`, `template-no-hints`,
  `template-no-original` and `template-minimal`. `--system-template` switches
  between the mutation-testing-expert system prompt and a minimal one.
- **RQ5 (model comparison).** `--provider openai --base-url …` points at any
  OpenAI-compatible endpoint, which is how the paper's mix of open and
  proprietary models is reproduced.
- **RQ1/RQ6 (mutant counts and cost).** `mutants.json` carries the same
  breakdown as Table 2 — prompts, candidates, invalid, identical, duplicate,
  mutants — plus prompt and completion token counts.

## Known limitations

- **Every mutant runs the full suite.** There is no coverage-based test
  selection, which is the largest speedup still available: running only the
  tests that execute the mutated line. On a real codebase, start with
  `--limit`.
- **Equivalent mutants are not detected.** The paper found roughly 20% of
  surviving mutants to be equivalent to the original code, and proposes
  AST-based filtering of common patterns as future work. The same applies here;
  surviving mutants need human review.
- **Placeholder locations are fixed.** As in the paper, the scheme is a
  compromise between mutant count and usefulness. `--kinds` widens or narrows
  it, but there is no per-project learning.
