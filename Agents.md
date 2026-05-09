# Agent Contract (Codex)

## Goal
Implement a runnable continual learning pipeline:
- CLEAR streaming protocol: train on bucket i, evaluate on bucket i+1
- Replay buffer baseline
- iNeMo-inspired proxy: prototype memory + latent partition loss
- Metrics: next-domain accuracy (NDA); optional forgetting via shadow holdout

## Hard rules
- Keep `python -m src.run` runnable at all times.
- Do not change YAML keys or artifact filenames without updating README and run.py accordingly.
- Prefer minimal dependencies; do not add heavy frameworks unless asked.

## Deliverables
- runs/<RUN_ID>/metrics_streaming.csv with NDA per step
- run_summary.json + config_resolved.yaml
- Optional metrics_shadow.csv when shadow_holdout_ratio > 0