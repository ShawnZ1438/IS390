import csv
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_streaming_metrics(path: str) -> Dict[str, List[float]]:
    steps: List[int] = []
    nda: List[float] = []
    mean_nda: List[float] = []
    train_buckets: List[str] = []
    test_buckets: List[str] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            nda.append(float(row["nda_on_next_bucket"]))
            mean_nda.append(float(row["mean_nda_so_far"]))
            train_buckets.append(row["train_bucket"])
            test_buckets.append(row["test_bucket"])

    return {
        "steps": steps,
        "nda": nda,
        "mean_nda": mean_nda,
        "train_buckets": train_buckets,
        "test_buckets": test_buckets,
    }


def read_shadow_metrics(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "after_step": int(row["after_step"]),
                    "bucket_eval": row["bucket_eval"],
                    "shadow_acc": float(row["shadow_acc"]),
                    "shadow_forgetting": float(row["shadow_forgetting"]),
                }
            )
    return rows


def summarize_shadow_by_step(rows: Sequence[Dict[str, float]]) -> Dict[str, List[float]]:
    grouped_acc: Dict[int, List[float]] = defaultdict(list)
    grouped_forgetting: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        step = int(row["after_step"])
        grouped_acc[step].append(float(row["shadow_acc"]))
        grouped_forgetting[step].append(float(row["shadow_forgetting"]))

    steps = sorted(grouped_acc.keys())
    mean_acc = [sum(grouped_acc[s]) / len(grouped_acc[s]) for s in steps]
    mean_forgetting = [
        sum(grouped_forgetting[s]) / len(grouped_forgetting[s]) for s in steps
    ]
    max_forgetting = [max(grouped_forgetting[s]) for s in steps]
    return {
        "steps": steps,
        "mean_acc": mean_acc,
        "mean_forgetting": mean_forgetting,
        "max_forgetting": max_forgetting,
    }


def plot_streaming_metrics(path: str, out_png: str, title: str = None) -> None:
    data = read_streaming_metrics(path)
    plt.figure(figsize=(7, 4.25))
    plt.plot(data["steps"], data["nda"], marker="o", linewidth=2, label="NDA@step")
    plt.plot(
        data["steps"],
        data["mean_nda"],
        marker="x",
        linewidth=2,
        label="Mean NDA so far",
    )
    plt.xlabel("Streaming step i (train i, test i+1)")
    plt.ylabel("Accuracy")
    plt.title(title or "Next-Domain Accuracy (NDA)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_shadow_metrics(path: str, out_png: str, title: str = None) -> None:
    summary = summarize_shadow_by_step(read_shadow_metrics(path))
    if not summary["steps"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.25))
    axes[0].plot(
        summary["steps"],
        summary["mean_acc"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    axes[0].set_title("Mean Shadow Accuracy")
    axes[0].set_xlabel("After training step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25, linewidth=0.5)

    axes[1].plot(
        summary["steps"],
        summary["mean_forgetting"],
        marker="o",
        linewidth=2,
        color="#d62728",
        label="Mean forgetting",
    )
    axes[1].plot(
        summary["steps"],
        summary["max_forgetting"],
        marker="x",
        linewidth=2,
        color="#9467bd",
        label="Max forgetting",
    )
    axes[1].set_title("Shadow Forgetting")
    axes[1].set_xlabel("After training step")
    axes[1].set_ylabel("Accuracy drop")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(alpha=0.25, linewidth=0.5)
    axes[1].legend()

    fig.suptitle(title or "Shadow Holdout Metrics")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
