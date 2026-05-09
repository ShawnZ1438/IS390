# IS390 Delivery Plan

Date: 2026-05-08

## Objective

Produce a delivery-ready continual-learning package that:

- keeps `python -m src.run` runnable,
- evaluates the CLEAR streaming protocol (`train i`, `test i+1`),
- compares the four core ablations,
- emits plots and summaries directly from the repo,
- includes a written report tied to exact run IDs.

## Execution Plan

1. Benchmark repair: completed
   - Replaced the previous random-noise synthetic fallback with a structured synthetic CLEAR proxy that has class signal plus bucket-wise domain shift.
   - Standardized the experiment configs onto the same dataset and enabled `shadow_holdout_ratio`.

2. Pipeline hardening: completed
   - Fixed evaluation to use sample-weighted accuracy.
   - Made prototype-only mode apply a real prototype alignment term.
   - Added automatic `nda.png`, `shadow.png`, and richer `run_summary.json` outputs.

3. Suite automation: completed
   - Added `scripts/run_ablation_suite.py`.
   - Added `scripts/aggregate_suite.py` for suite tables and figures.

4. Experiments and packaging: completed
   - Ran the four ablations on the delivery benchmark.
   - Generated suite-level plots and tables in this directory.
   - Wrote `REPORT.md` with findings, limitations, and reproduction instructions.

## Reproduction

Single run:

```powershell
python -m src.run --config configs/base.yaml
```

Full suite:

```powershell
python scripts/run_ablation_suite.py --manifest_out deliverables/20260508_delivery/ablation_manifest.json --aggregate_out_dir deliverables/20260508_delivery
```

## Artifact Index

- `ablation_manifest.json`: mapping from config to run ID and run directory
- `suite_summary.csv`: machine-readable results table
- `suite_summary.md`: human-readable summary table
- `nda_by_step.png`: cross-ablation NDA comparison
- `mean_nda_bar.png`: mean NDA comparison
- `shadow_mean_accuracy.png`: shadow holdout accuracy comparison
- `shadow_mean_forgetting.png`: shadow forgetting comparison
- `REPORT.md`: final delivery narrative
