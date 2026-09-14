# Example project

A tiny package with an intentionally incomplete test suite, used to demonstrate
`llmorpheus-py` end to end without spending any tokens:

```bash
cd examples/pricing
python -m llmorpheus all --provider mock --src sample_pkg
```

The mock provider produces mutants offline, so this runs anywhere. Swap in
`--provider openai --model gpt-4o-mini` (with `OPENAI_API_KEY` set) to see what
a real model suggests.
