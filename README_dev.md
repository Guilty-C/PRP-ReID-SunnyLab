# Developer Notes

## Prompt Augmentation (CLI)

The prompt tuner exposes augmentation controls through `tools/prompt_tuner/agent.py`.
With defaults (`--ensemble 1`, `--neg-rate 0.0`, deterministic enabled), behaviour
matches the legacy flow.

| Flag | Default | Description |
| --- | --- | --- |
| `--ensemble` | `1` | Number of prompt variants to emit per base prompt. |
| `--neg-rate` | `0.0` | Probability of appending one safety-focused negative clause. |
| `--seed` | _unset_ | Optional seed applied when deterministic augmentation is enabled. |
| `--no-deterministic-augment` | off | Disable deterministic behaviour even if a seed is supplied. |
| `--log-level` | `INFO` | Logging verbosity for the CLI. |

Example invocation:

```bash
python tools/prompt_tuner/agent.py \
  --rounds 1 \
  --ensemble 3 \
  --neg-rate 0.25 \
  --seed 123 \
  --log-level DEBUG
```
