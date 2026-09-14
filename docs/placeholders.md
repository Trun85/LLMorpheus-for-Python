# The placeholder scheme

Where a placeholder may be introduced, and therefore what the LLM is asked
about. This is the Python counterpart of Figure 6 of the paper, which targets
loop headers, branch conditions, and the receiver and arguments of calls.

Inspect what a scheme would produce before spending anything:

```bash
python -m llmorpheus locations --src src
python -m llmorpheus locations --src src --kinds all --json
```

## Default scheme

Selected by default, or explicitly with `--kinds default`. `<P>` marks the
placeholder.

| Kind | Example | Paper's analogue |
| --- | --- | --- |
| `if-condition` | `if <P>:` — also covers `elif` | `if (<P>)` |
| `while-condition` | `while <P>:` | `while (<P>)` |
| `ternary-condition` | `a if <P> else b` | — |
| `assert-condition` | `assert <P>` | — |
| `match-subject` | `match <P>:` (Python 3.10+) | `switch (<P>)` |
| `for-target` | `for <P> in xs:` | `for (<P> of obj)` |
| `for-iter` | `for x in <P>:` | `for (o of <P>)` |
| `comprehension-iter` | `[x for x in <P>]` | loop header |
| `comprehension-condition` | `[x for x in xs if <P>]` | loop header |
| `call-callee` | `<P>(x, y)` | `<P>(x, y)` |
| `call-argument` | `f(<P>, y)` | `a.m(<P>, y)` |
| `call-keyword-argument` | `f(x, key=<P>)` | — |
| `call-arguments` | `f(<P>)` — the whole argument list | `a.m(<P>)` |

A call with exactly one positional argument does not also get a
`call-arguments` location, since it would duplicate `call-argument`.

## Extended scheme

Added by `--kinds all`. These find more, but multiply the mutant count and
therefore the runtime.

| Kind | Example |
| --- | --- |
| `return-value` | `return <P>` |
| `assign-value` | `x = <P>`, `x += <P>`, `x: int = <P>` |
| `subscript-index` | `xs[<P>]` (slices are excluded) |
| `with-context` | `with <P> as f:` |
| `except-type` | `except <P>:` |

## Selecting kinds

```bash
--kinds default                       # the table above (default)
--kinds all                           # default + extended
--kinds if-condition,call-argument    # just these
--kinds if-condition --kinds for-iter # repeatable
```

## What is never a location

- **Anything inside an f-string.** Before Python 3.12 those nodes carry
  positions that do not match the source. The f-string as a whole is still
  eligible when it is an argument or a condition.
- **Fragments that are too large.** Over `--max-fragment-chars` (240) or
  `--max-fragment-lines` (8), to keep prompts and mutants sane.
- **Files that do not parse**, and files excluded by the selector: `tests/`,
  `test_*.py`, `*_test.py`, `conftest.py`, `setup.py`, `_version.py`,
  `__main__.py`, virtualenvs, caches and build directories. Override with
  `--include` / `--exclude`.
