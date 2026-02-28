import os
import csv
import json
import argparse
import yaml
from src.utils import set_seed, ensure_dir, now_ts
from src.data.clear_dataset import ClearDataConfig, make_bucket_loaders
from src.data.synthetic_clear import SyntheticSpec, generate_synthetic
from src.models.backbone import build_backbone
from src.models.head import LinearHead
from src.training.streaming_trainer import (
    StreamingTrainer, TrainConfig, InemoLikeConfig
)
from src.memory.replay_buffer import ReplayConfig
from src.training.metrics import ForgettingLog


def _as_bool(x):
    return bool(x)


def _maybe_generate_synthetic(cfg) -> None:
    syn = cfg.get("synthetic", {})
    if not syn.get("enabled", False):
        return
    data_root = cfg["data"].get("root", "")
    if data_root and os.path.isdir(data_root):
        return
    out_root = syn.get("out_root", "./data/synth_CLEAR")
    spec = SyntheticSpec(
        out_root=out_root,
        layout=syn.get("layout", "imagefolder"),
        num_buckets=int(syn.get("num_buckets", 6)),
        num_classes=int(syn.get("num_classes", 11)),
        imgs_per_class_per_bucket=int(syn.get("imgs_per_class_per_bucket", 20)),
        image_size=int(cfg["data"].get("image_size", 224)),
        seed=int(syn.get("seed", 123)),
    )
    cfg["data"]["root"] = generate_synthetic(spec)


def _infer_num_classes(cfg, class_to_idx):
    v = cfg["model"].get("num_classes", "auto")
    if isinstance(v, str) and v.lower() == "auto":
        return len(class_to_idx)
    return int(v)


def main(cfg_path: str = "configs/base.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))
    _maybe_generate_synthetic(cfg)
    out_dir = ensure_dir(os.path.join(cfg["logging"]["out_dir"], now_ts()))
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # Data
    dcfg = ClearDataConfig(
        root=cfg["data"]["root"],
        layout=cfg["data"].get("layout", "auto"),
        bucket_glob=cfg["data"].get("bucket_glob", "bucket_*"),
        manifest_name=cfg["data"].get("manifest_name", "labels.csv"),
        image_size=int(cfg["data"].get("image_size", 224)),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        batch_size=int(cfg["train"]["batch_size"]),
    )
    class_map_path = (
        os.path.join(out_dir, "class_map.json")
        if cfg["logging"].get("save_class_map", True)
        else None
    )
    buckets, class_to_idx, train_loaders, eval_full_loaders, shadow_loaders = (
        make_bucket_loaders(
            dcfg,
            shadow_holdout_ratio=float(cfg["eval"].get("shadow_holdout_ratio", 0.0)),
            seed=int(cfg.get("seed", 42)),
            save_class_map_path=class_map_path,
        )
    )
    num_classes = _infer_num_classes(cfg, class_to_idx)

    # Model
    backbone, feat_dim = build_backbone(
        cfg["model"]["backbone"], int(cfg["model"]["feature_dim"])
    )
    head = LinearHead(feat_dim, num_classes)

    # Trainer configs
    tcfg = TrainConfig(
        device=cfg["train"].get("device", "cuda"),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
        epochs_per_bucket=int(cfg["train"].get("epochs_per_bucket", 1)),
        batch_size=int(cfg["train"].get("batch_size", 64)),
        log_every=int(cfg["train"].get("log_every", 50)),
    )
    rcfg = ReplayConfig(
        buffer_size=int(cfg["replay"].get("buffer_size", 2000)),
        strategy=cfg["replay"].get("strategy", "reservoir"),
        alpha=float(cfg["replay"].get("alpha", 1.0)),
        per_class_cap=cfg["replay"].get("per_class_cap", 50),
        seed=int(cfg.get("seed", 42)),
    )

    inemo_raw = cfg.get("inemo_like", {})
    inemo_cfg = InemoLikeConfig(
        enabled=_as_bool(inemo_raw.get("enabled", True)),
        prototypes_enabled=_as_bool(
            inemo_raw.get("prototypes", {}).get("enabled", True)
        ),
        proto_momentum=float(
            inemo_raw.get("prototypes", {}).get("momentum", 0.9)
        ),
        latent_partition_enabled=_as_bool(
            inemo_raw.get("latent_partition", {}).get("enabled", True)
        ),
        partition_strength=float(
            inemo_raw.get("latent_partition", {}).get("strength", 0.05)
        ),
        partition_margin=float(
            inemo_raw.get("latent_partition", {}).get("margin", 0.2)
        ),
    )
    trainer = StreamingTrainer(
        backbone=backbone,
        head=head,
        feature_dim=feat_dim,
        num_classes=num_classes,
        train_cfg=tcfg,
        replay_cfg=rcfg,
        inemo_cfg=inemo_cfg,
        replay_ratio=(
            float(cfg["replay"].get("ratio", 0.3))
            if cfg["replay"].get("enabled", True)
            else 0.0
        ),
    )

    # Logging
    metrics_path = os.path.join(out_dir, "metrics_streaming.csv")
    f_shadow = ForgettingLog()
    shadow_path = os.path.join(out_dir, "metrics_shadow.csv")

    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "train_bucket",
                "test_bucket",
                "nda_on_next_bucket",
                "mean_nda_so_far",
            ]
        )
        nda_vals = []
        # Streaming: train on i, test on i+1
        for i in range(len(train_loaders) - 1):
            trainer.train_one_bucket(train_loaders[i], bucket_idx=i)
            nda = trainer.eval_loader(eval_full_loaders[i + 1])
            nda_vals.append(nda)
            mean_nda = sum(nda_vals) / len(nda_vals)
            w.writerow(
                [
                    i,
                    buckets[i].name,
                    buckets[i + 1].name,
                    f"{nda:.6f}",
                    f"{mean_nda:.6f}",
                ]
            )
            print(
                f"[step {i}] train={buckets[i].name} "
                f"test_next={buckets[i+1].name} NDA={nda:.4f}"
            )

    # Optional: "shadow" forgetting metrics (requires holdout_ratio > 0)
    shadow_hold = float(cfg["eval"].get("shadow_holdout_ratio", 0.0))
    if shadow_hold > 0 and cfg["eval"].get("compute_forgetting", True):
        with open(shadow_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["after_step", "bucket_eval", "shadow_acc", "shadow_forgetting"]
            )
            for i in range(len(train_loaders) - 1):
                # After training steps, evaluate all prior shadow sets up to i
                for j in range(i + 1):
                    if shadow_loaders[j] is None:
                        continue
                    acc = trainer.eval_loader(shadow_loaders[j])
                    forgetting = f_shadow.update_bucket(j, acc)
                    w.writerow(
                        [
                            i,
                            buckets[j].name,
                            f"{acc:.6f}",
                            f"{forgetting:.6f}",
                        ]
                    )

    # Save a compact run summary
    summary = {
        "out_dir": out_dir,
        "num_buckets": len(buckets),
        "num_classes": num_classes,
        "metrics_streaming_csv": metrics_path,
        "metrics_shadow_csv": shadow_path if (shadow_hold > 0) else None,
        "class_map_json": class_map_path,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary:", os.path.join(out_dir, "run_summary.json"))




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config file.",
    )
    args = ap.parse_args()
    main(cfg_path=args.config)
