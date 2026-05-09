import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.reporting import plot_shadow_metrics, plot_streaming_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streaming_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument(
        "--shadow_csv",
        default=None,
        help="Optional shadow metrics CSV to plot alongside the streaming plot.",
    )
    ap.add_argument(
        "--shadow_out_png",
        default=None,
        help="Output path for the shadow holdout plot when --shadow_csv is provided.",
    )
    args = ap.parse_args()

    plot_streaming_metrics(args.streaming_csv, args.out_png)
    if args.shadow_csv:
        if not args.shadow_out_png:
            raise ValueError("--shadow_out_png is required when --shadow_csv is set.")
        plot_shadow_metrics(args.shadow_csv, args.shadow_out_png)


if __name__ == "__main__":
    main()
