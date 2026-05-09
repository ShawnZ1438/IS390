import csv
import argparse
import matplotlib.pyplot as plt

def read_streaming(path):
    steps, nda, mean_nda = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            nda.append(float(row["nda_on_next_bucket"]))
            mean_nda.append(float(row["mean_nda_so_far"]))
    return steps, nda, mean_nda

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streaming_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    steps, nda, mean_nda = read_streaming(args.streaming_csv)
    plt.figure()
    plt.plot(steps, nda, marker="o", label="NDA@step")
    plt.plot(steps, mean_nda, marker="x", label="Mean NDA so far")
    plt.xlabel("Streaming step i (train i, test i+1)")
    plt.ylabel("Accuracy")
    plt.title("Next-Domain Accuracy (NDA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)

if __name__ == "__main__":
    main()