# IS390 - CLEAR Streaming CL + iNeMo-inspired Proxy

This repo implements a runnable continual-learning pipeline for the CLEAR-style
streaming protocol:

- Train on bucket `i`
- Evaluate on bucket `i + 1`
- Compare `finetune`, `replay`, `replay + prototypes`, and `full iNeMo-like proxy`

The default configs now point at a structured synthetic benchmark
(`./data/synth_CLEAR_delivery_v1`) that preserves class identity while applying
bucket-wise domain shift. If the dataset is missing, `python -m src.run`
generates it automatically.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate Synthetic CLEAR Buckets (optional)
```powershell
python -m src.data.synthetic_clear --out_dir ./data/synth_CLEAR_delivery_v1 --layout imagefolder --num_buckets 6 --num_classes 11 --imgs_per_class_per_bucket 40 --image_size 64 --seed 123
```

## Run
Default:
```powershell
python -m src.run --config configs/base.yaml
```

## Ablations
```powershell
python -m src.run --config configs/finetune_only.yaml
python -m src.run --config configs/replay_baseline.yaml
python -m src.run --config configs/replay_plus_prototypes.yaml
python -m src.run --config configs/full_inemo_proxy.yaml
```

## Outputs per run
`runs/<RUN_ID>/`

- `config_resolved.yaml`
- `metrics_streaming.csv` with columns: `step,train_bucket,test_bucket,nda_on_next_bucket,mean_nda_so_far`
- `metrics_shadow.csv` when `eval.shadow_holdout_ratio > 0`
- `run_summary.json`
- `class_map.json` when `logging.save_class_map: true`
- `nda.png` when `logging.make_plots: true`
- `shadow.png` when `logging.make_plots: true` and shadow metrics are enabled
- `model_final.pt` when `logging.save_model: true`

`run_summary.json` now includes aggregate NDA fields and shadow-summary fields so
suite-level reporting can be built without re-parsing everything manually.

## Plot a Single Run
```powershell
python scripts/plot_metrics.py --streaming_csv runs/<RUN_ID>/metrics_streaming.csv --out_png runs/<RUN_ID>/nda.png --shadow_csv runs/<RUN_ID>/metrics_shadow.csv --shadow_out_png runs/<RUN_ID>/shadow.png
```

## Run the Full Ablation Suite
```powershell
python scripts/run_ablation_suite.py --manifest_out deliverables/<STAMP>/ablation_manifest.json --aggregate_out_dir deliverables/<STAMP>
```

This runs the four core configs and writes:

- `deliverables/<STAMP>/ablation_manifest.json`
- `deliverables/<STAMP>/suite_summary.csv`
- `deliverables/<STAMP>/suite_summary.md`
- `deliverables/<STAMP>/nda_by_step.png`
- `deliverables/<STAMP>/mean_nda_bar.png`
- `deliverables/<STAMP>/shadow_mean_accuracy.png`
- `deliverables/<STAMP>/shadow_mean_forgetting.png`

## Latest Delivery

The current delivery note is in:

- [deliverables/20260508_delivery/REPORT.md](./deliverables/20260508_delivery/REPORT.md)
