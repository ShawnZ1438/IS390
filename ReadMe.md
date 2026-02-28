# IS390 - CLEAR Streaming CL + iNeMo-inspired Proxy

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate Synthetic CLEAR Buckets (optional)
```powershell
python -m src.data.synthetic_clear --out_dir ./data/synth_CLEAR --layout imagefolder --num_buckets 6 --num_classes 11 --imgs_per_class_per_bucket 20 --image_size 64 --seed 123
```

## Run
Default (uses synthetic fallback when `data.root` is empty or missing):
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
- `run_summary.json`
- `class_map.json` (when `logging.save_class_map: true`)
- `metrics_shadow.csv` (when `eval.shadow_holdout_ratio > 0`)

## Plot NDA
```powershell
python scripts/plot_metrics.py --streaming_csv runs/<RUN_ID>/metrics_streaming.csv --out_png runs/<RUN_ID>/nda.png
```
