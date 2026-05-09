import os
import argparse
from dataclasses import dataclass
from typing import Literal
import numpy as np
from PIL import Image

Layout = Literal["imagefolder", "manifest_csv"]

@dataclass
class SyntheticSpec:
    out_root: str
    layout: Layout
    num_buckets: int
    num_classes: int
    imgs_per_class_per_bucket: int
    image_size: int
    seed: int

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _rand_img(rng: np.random.Generator, size: int) -> Image.Image:
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def generate_synthetic(spec: SyntheticSpec) -> str:
    rng = np.random.default_rng(spec.seed)
    _ensure_dir(spec.out_root)

    for b in range(spec.num_buckets):
        bdir = os.path.join(spec.out_root, f"bucket_{b:03d}")
        _ensure_dir(bdir)

        if spec.layout == "imagefolder":
            for c in range(spec.num_classes):
                cdir = os.path.join(bdir, f"class_{c:03d}")
                _ensure_dir(cdir)
                for i in range(spec.imgs_per_class_per_bucket):
                    img = _rand_img(rng, spec.image_size)
                    img.save(os.path.join(cdir, f"img_{i:05d}.jpg"), quality=90)

        elif spec.layout == "manifest_csv":
            img_dir = os.path.join(bdir, "images")
            _ensure_dir(img_dir)
            man_path = os.path.join(bdir, "labels.csv")
            with open(man_path, "w") as f:
                f.write("filepath,label\n")
                for c in range(spec.num_classes):
                    lab = f"class_{c:03d}"
                    for i in range(spec.imgs_per_class_per_bucket):
                        fn = f"b{b:03d}_c{c:03d}_img{i:05d}.jpg"
                        rel = os.path.join("images", fn)
                        img = _rand_img(rng, spec.image_size)
                        img.save(os.path.join(img_dir, fn), quality=90)
                        f.write(f"{rel},{lab}\n")
        else:
            raise ValueError(spec.layout)

    return spec.out_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layout", default="imagefolder", choices=["imagefolder", "manifest_csv"])
    ap.add_argument("--num_buckets", type=int, default=6)
    ap.add_argument("--num_classes", type=int, default=11)
    ap.add_argument("--imgs_per_class_per_bucket", type=int, default=20)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    spec = SyntheticSpec(
        out_root=args.out_dir,
        layout=args.layout,
        num_buckets=args.num_buckets,
        num_classes=args.num_classes,
        imgs_per_class_per_bucket=args.imgs_per_class_per_bucket,
        image_size=args.image_size,
        seed=args.seed,
    )
    out = generate_synthetic(spec)
    print(out)

if __name__ == "__main__":
    main()