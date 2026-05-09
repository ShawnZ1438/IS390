import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.reporting import read_shadow_metrics, read_streaming_metrics, summarize_shadow_by_step


def _load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["runs"]


def _summary_row(label: str, config_path: str, run_dir: str, summary: Dict) -> Dict[str, object]:
    shadow = summary.get("shadow_summary") or {}
    return {
        "label": label,
        "config_path": config_path,
        "run_id": summary.get("run_id"),
        "run_dir": run_dir,
        "mean_nda": summary.get("mean_nda"),
        "best_nda": summary.get("best_nda"),
        "last_nda": summary.get("last_nda"),
        "last_mean_shadow_acc": shadow.get("last_mean_shadow_acc"),
        "last_mean_shadow_forgetting": shadow.get("last_mean_shadow_forgetting"),
        "max_shadow_forgetting": shadow.get("max_shadow_forgetting"),
    }


def _write_summary_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "config_path",
                "run_id",
                "run_dir",
                "mean_nda",
                "best_nda",
                "last_nda",
                "last_mean_shadow_acc",
                "last_mean_shadow_forgetting",
                "max_shadow_forgetting",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_md(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w") as f:
        f.write("# Ablation Summary\n\n")
        f.write("| Label | Mean NDA | Best NDA | Last NDA | Last Mean Shadow Acc | Last Mean Forgetting | Max Forgetting | Run ID |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            f.write(
                "| {label} | {mean_nda:.4f} | {best_nda:.4f} | {last_nda:.4f} | {last_mean_shadow_acc} | {last_mean_shadow_forgetting} | {max_shadow_forgetting} | {run_id} |\n".format(
                    label=row["label"],
                    mean_nda=float(row["mean_nda"]),
                    best_nda=float(row["best_nda"]),
                    last_nda=float(row["last_nda"]),
                    last_mean_shadow_acc=(
                        f"{float(row['last_mean_shadow_acc']):.4f}"
                        if row["last_mean_shadow_acc"] is not None
                        else "-"
                    ),
                    last_mean_shadow_forgetting=(
                        f"{float(row['last_mean_shadow_forgetting']):.4f}"
                        if row["last_mean_shadow_forgetting"] is not None
                        else "-"
                    ),
                    max_shadow_forgetting=(
                        f"{float(row['max_shadow_forgetting']):.4f}"
                        if row["max_shadow_forgetting"] is not None
                        else "-"
                    ),
                    run_id=row["run_id"],
                )
            )


def _plot_nda_by_step(path: str, run_data: Sequence[Dict[str, object]]) -> None:
    plt.figure(figsize=(8, 4.5))
    for item in run_data:
        plt.plot(
            item["streaming"]["steps"],
            item["streaming"]["nda"],
            marker="o",
            linewidth=2,
            label=item["label"],
        )
    plt.xlabel("Streaming step")
    plt.ylabel("NDA")
    plt.title("NDA by Streaming Step")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_mean_nda_bar(path: str, rows: Sequence[Dict[str, object]]) -> None:
    labels = [row["label"] for row in rows]
    values = [float(row["mean_nda"]) for row in rows]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"])
    plt.ylabel("Mean NDA")
    plt.title("Mean Next-Domain Accuracy by Ablation")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.25, linewidth=0.5)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_shadow_metric(path: str, run_data: Sequence[Dict[str, object]], key: str, title: str, ylabel: str) -> None:
    eligible = [item for item in run_data if item.get("shadow_summary")]
    if not eligible:
        return

    plt.figure(figsize=(8, 4.5))
    for item in eligible:
        plt.plot(
            item["shadow_summary"]["steps"],
            item["shadow_summary"][key],
            marker="o",
            linewidth=2,
            label=item["label"],
        )
    plt.xlabel("After training step")
    plt.ylabel(ylabel)
    plt.title(title)
    if key == "mean_acc":
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_rows = _load_manifest(args.manifest)
    rows: List[Dict[str, object]] = []
    run_data: List[Dict[str, object]] = []

    for item in manifest_rows:
        with open(item["summary_path"], "r") as f:
            summary = json.load(f)
        streaming = read_streaming_metrics(summary["metrics_streaming_csv"])
        shadow_summary = None
        shadow_csv = summary.get("metrics_shadow_csv")
        if shadow_csv and os.path.isfile(shadow_csv):
            shadow_summary = summarize_shadow_by_step(read_shadow_metrics(shadow_csv))

        rows.append(
            _summary_row(
                label=item["label"],
                config_path=item["config_path"],
                run_dir=item["run_dir"],
                summary=summary,
            )
        )
        run_data.append(
            {
                "label": item["label"],
                "streaming": streaming,
                "shadow_summary": shadow_summary,
            }
        )

    rows.sort(key=lambda row: float(row["mean_nda"]), reverse=True)

    _write_summary_csv(os.path.join(args.out_dir, "suite_summary.csv"), rows)
    _write_summary_md(os.path.join(args.out_dir, "suite_summary.md"), rows)
    _plot_nda_by_step(os.path.join(args.out_dir, "nda_by_step.png"), run_data)
    _plot_mean_nda_bar(os.path.join(args.out_dir, "mean_nda_bar.png"), rows)
    _plot_shadow_metric(
        os.path.join(args.out_dir, "shadow_mean_accuracy.png"),
        run_data,
        key="mean_acc",
        title="Mean Shadow Accuracy by Step",
        ylabel="Accuracy",
    )
    _plot_shadow_metric(
        os.path.join(args.out_dir, "shadow_mean_forgetting.png"),
        run_data,
        key="mean_forgetting",
        title="Mean Shadow Forgetting by Step",
        ylabel="Accuracy drop",
    )

    with open(os.path.join(args.out_dir, "suite_manifest.json"), "w") as f:
        json.dump({"runs": manifest_rows}, f, indent=2)
    print(f"Saved aggregate outputs to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
