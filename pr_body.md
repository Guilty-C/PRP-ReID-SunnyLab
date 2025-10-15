### Summary
- Introduces `tools/prompt_tuner/augment.py` with `AugmentConfig` and `augment_prompts`.
- Adds CLI flags: `--ensemble`, `--neg-rate`, `--seed`, `--no-deterministic-augment`, `--log-level`.
- Updates `agent.py` to apply augmentation after base prompts are built.
- Adds minimal smoke and unit tests; updates `README_dev.md`.

### Backward Compatibility
- Defaults (`ensemble=1`, `neg_rate=0.0`) preserve pre-change behavior.

### Reviewer Checklist
- [ ] `--help` shows new flags with clear descriptions.
- [ ] Defaults do not alter previous results.
- [ ] Determinism: seeding is honored; no random behavior at import.
- [ ] Logging replaces prints; levels are sensible.
- [ ] Smoke and unit tests are sufficient and fast.

### Notes
No third-party deps added; Windows paths remain compatible.
