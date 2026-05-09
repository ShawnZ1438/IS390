import argparse
import colorsys
import math
import os
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

Layout = Literal["imagefolder", "manifest_csv"]

_CLASS_SPECS: Sequence[Tuple[str, str, float]] = (
    ("circle", "horizontal", 0.02),
    ("square", "horizontal", 0.10),
    ("triangle", "horizontal", 0.18),
    ("diamond", "horizontal", 0.26),
    ("circle", "vertical", 0.36),
    ("square", "vertical", 0.44),
    ("triangle", "vertical", 0.52),
    ("diamond", "vertical", 0.60),
    ("circle", "diagonal", 0.70),
    ("square", "diagonal", 0.78),
    ("triangle", "ring", 0.86),
    ("diamond", "ring", 0.94),
)


@dataclass
class SyntheticSpec:
    out_root: str
    layout: Layout
    num_buckets: int
    num_classes: int
    imgs_per_class_per_bucket: int
    image_size: int
    seed: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rgb_from_hsv(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(
        int(round((1.0 - t) * av + t * bv))
        for av, bv in zip(a, b)
    )


def _bucket_palette(bucket_idx: int, num_buckets: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    phase = bucket_idx / max(1, num_buckets - 1)
    bg_top = _rgb_from_hsv(0.56 + 0.23 * phase, 0.22 + 0.05 * phase, 0.96)
    bg_bottom = _rgb_from_hsv(0.08 + 0.32 * phase, 0.28, 0.74 - 0.05 * phase)
    accent = _rgb_from_hsv(0.20 + 0.45 * phase, 0.42, 0.92)
    return bg_top, bg_bottom, accent


def _make_background(
    rng: np.random.Generator,
    size: int,
    bucket_idx: int,
    num_buckets: int,
) -> Tuple[Image.Image, Tuple[int, int, int]]:
    bg_top, bg_bottom, accent = _bucket_palette(bucket_idx, num_buckets)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None]

    top = np.asarray(bg_top, dtype=np.float32)[None, None, :]
    bottom = np.asarray(bg_bottom, dtype=np.float32)[None, None, :]
    bg = top * (1.0 - y) + bottom * y

    phase = bucket_idx / max(1, num_buckets - 1)
    wave = 14.0 * np.sin(2.0 * math.pi * ((2.0 + 0.35 * bucket_idx) * x + (1.1 + phase) * y))
    wave += 10.0 * np.cos(2.0 * math.pi * ((1.5 + phase) * x - (2.4 + 0.2 * bucket_idx) * y))
    vignette = ((x - 0.5) ** 2 + (y - 0.5) ** 2) * (36.0 + 18.0 * phase)
    noise = rng.normal(0.0, 4.0 + 6.0 * phase, size=(size, size, 1))

    bg = bg + wave - vignette + noise
    bg = np.clip(bg, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(bg, mode="RGB"), accent


def _shape_bbox(
    rng: np.random.Generator,
    size: int,
) -> Tuple[int, int, int, int]:
    cx = int(size * 0.5 + rng.integers(-size // 14, size // 14 + 1))
    cy = int(size * 0.53 + rng.integers(-size // 12, size // 12 + 1))
    radius = int(size * (0.24 + float(rng.uniform(-0.025, 0.025))))
    return cx - radius, cy - radius, cx + radius, cy + radius


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape_kind: str,
    bbox: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int,
) -> None:
    x0, y0, x1, y1 = bbox
    if shape_kind == "circle":
        draw.ellipse(bbox, fill=fill, outline=outline, width=width)
        return
    if shape_kind == "square":
        draw.rounded_rectangle(bbox, radius=max(2, width), fill=fill, outline=outline, width=width)
        return
    if shape_kind == "triangle":
        points = [
            ((x0 + x1) // 2, y0),
            (x1, y1),
            (x0, y1),
        ]
        draw.polygon(points, fill=fill, outline=outline)
        return
    if shape_kind == "diamond":
        points = [
            ((x0 + x1) // 2, y0),
            (x1, (y0 + y1) // 2),
            ((x0 + x1) // 2, y1),
            (x0, (y0 + y1) // 2),
        ]
        draw.polygon(points, fill=fill, outline=outline)
        return
    raise ValueError(f"Unsupported shape kind: {shape_kind}")


def _draw_motif(
    draw: ImageDraw.ImageDraw,
    motif_kind: str,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    x0, y0, x1, y1 = bbox
    inner_pad = max(3, (x1 - x0) // 5)
    xi0 = x0 + inner_pad
    yi0 = y0 + inner_pad
    xi1 = x1 - inner_pad
    yi1 = y1 - inner_pad

    if motif_kind == "horizontal":
        for frac in (0.25, 0.5, 0.75):
            y = int(round(yi0 + frac * (yi1 - yi0)))
            draw.line((xi0, y, xi1, y), fill=color, width=width)
        return
    if motif_kind == "vertical":
        for frac in (0.25, 0.5, 0.75):
            x = int(round(xi0 + frac * (xi1 - xi0)))
            draw.line((x, yi0, x, yi1), fill=color, width=width)
        return
    if motif_kind == "diagonal":
        span = yi1 - yi0
        offsets = (-span // 4, 0, span // 4)
        for delta in offsets:
            draw.line((xi0, yi1 + delta, xi1, yi0 + delta), fill=color, width=width)
        return
    if motif_kind == "ring":
        draw.ellipse((xi0, yi0, xi1, yi1), outline=color, width=width)
        return
    raise ValueError(f"Unsupported motif kind: {motif_kind}")


def _apply_bucket_overlay(
    img: Image.Image,
    accent: Tuple[int, int, int],
    bucket_idx: int,
    num_buckets: int,
) -> Image.Image:
    size = img.size[0]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    phase = bucket_idx / max(1, num_buckets - 1)
    stripe_alpha = 24 + int(round(34 * phase))
    stripe_gap = 12 + 2 * (bucket_idx % 3)
    for offset in range(-size, size * 2, stripe_gap):
        draw.line(
            (offset, 0, offset - size, size),
            fill=accent + (stripe_alpha,),
            width=2,
        )

    if bucket_idx >= 2:
        haze = _mix_rgb(accent, (255, 255, 255), 0.55)
        draw.rounded_rectangle(
            (
                int(size * 0.08),
                int(size * 0.08),
                int(size * 0.92),
                int(size * 0.92),
            ),
            radius=max(6, size // 12),
            outline=haze + (28 + 10 * bucket_idx,),
            width=max(2, size // 28),
        )

    if bucket_idx >= 3:
        occlusion_color = _mix_rgb(accent, (0, 0, 0), 0.35)
        band_w = max(5, size // 10)
        x0 = int(size * (0.15 + 0.12 * (bucket_idx % 3)))
        draw.rectangle((x0, 0, x0 + band_w, size), fill=occlusion_color + (32 + 6 * bucket_idx,))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _finalize_image(
    rng: np.random.Generator,
    img: Image.Image,
    bg_fill: Tuple[int, int, int],
    bucket_idx: int,
    num_buckets: int,
) -> Image.Image:
    phase = bucket_idx / max(1, num_buckets - 1)
    angle = -14.0 + 28.0 * phase + float(rng.normal(0.0, 2.5))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg_fill)

    color_factor = 0.9 + 0.18 * math.cos(math.pi * phase)
    contrast_factor = 0.95 + 0.2 * math.sin(math.pi * phase)
    brightness_factor = 0.94 + 0.1 * math.cos(0.5 * math.pi * phase)
    img = ImageEnhance.Color(img).enhance(color_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    blur_radius = 0.1 + 0.8 * phase
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0.0, 5.0 + 9.0 * phase, size=arr.shape)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _class_style(class_idx: int) -> Tuple[str, str, Tuple[int, int, int]]:
    shape_kind, motif_kind, hue = _CLASS_SPECS[class_idx % len(_CLASS_SPECS)]
    color = _rgb_from_hsv(hue, 0.7, 0.92)
    return shape_kind, motif_kind, color


def _draw_corner_markers(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    class_idx: int,
    color: Tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = bbox
    dot_r = max(2, (x1 - x0) // 16)
    markers = 1 + (class_idx % 3)
    for i in range(markers):
        offset = i * (dot_r * 3)
        cx = x0 + dot_r * 2 + offset
        cy = y0 + dot_r * 2
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=color)


def _render_sample(
    rng: np.random.Generator,
    size: int,
    class_idx: int,
    bucket_idx: int,
    num_buckets: int,
) -> Image.Image:
    img, accent = _make_background(rng, size, bucket_idx, num_buckets)
    bg_fill = tuple(int(v) for v in np.asarray(img).mean(axis=(0, 1)))
    draw = ImageDraw.Draw(img)

    bbox = _shape_bbox(rng, size)
    shape_kind, motif_kind, class_color = _class_style(class_idx)
    fill = _mix_rgb(class_color, accent, 0.18 + 0.12 * math.sin(bucket_idx + class_idx))
    outline = _mix_rgb(fill, (15, 18, 24), 0.45)
    motif_color = _mix_rgb((255, 255, 255), accent, 0.25)

    outer_width = max(2, size // 24)
    motif_width = max(2, size // 28)
    _draw_shape(draw, shape_kind, bbox, fill=fill, outline=outline, width=outer_width)
    _draw_motif(draw, motif_kind, bbox, color=motif_color, width=motif_width)
    _draw_corner_markers(draw, bbox, class_idx, color=outline)

    img = _apply_bucket_overlay(img, accent, bucket_idx, num_buckets)
    return _finalize_image(rng, img, bg_fill, bucket_idx, num_buckets)


def generate_synthetic(spec: SyntheticSpec) -> str:
    root_rng = np.random.default_rng(spec.seed)
    _ensure_dir(spec.out_root)

    for b in range(spec.num_buckets):
        bdir = os.path.join(spec.out_root, f"bucket_{b:03d}")
        _ensure_dir(bdir)

        if spec.layout == "imagefolder":
            for c in range(spec.num_classes):
                cdir = os.path.join(bdir, f"class_{c:03d}")
                _ensure_dir(cdir)
                for i in range(spec.imgs_per_class_per_bucket):
                    sample_seed = int(root_rng.integers(0, 2**31 - 1))
                    img = _render_sample(
                        np.random.default_rng(sample_seed),
                        spec.image_size,
                        c,
                        b,
                        spec.num_buckets,
                    )
                    img.save(os.path.join(cdir, f"img_{i:05d}.jpg"), quality=92)

        elif spec.layout == "manifest_csv":
            img_dir = os.path.join(bdir, "images")
            _ensure_dir(img_dir)
            man_path = os.path.join(bdir, "labels.csv")
            with open(man_path, "w", newline="") as f:
                f.write("filepath,label\n")
                for c in range(spec.num_classes):
                    lab = f"class_{c:03d}"
                    for i in range(spec.imgs_per_class_per_bucket):
                        sample_seed = int(root_rng.integers(0, 2**31 - 1))
                        img = _render_sample(
                            np.random.default_rng(sample_seed),
                            spec.image_size,
                            c,
                            b,
                            spec.num_buckets,
                        )
                        fn = f"b{b:03d}_c{c:03d}_img{i:05d}.jpg"
                        rel = os.path.join("images", fn)
                        img.save(os.path.join(img_dir, fn), quality=92)
                        f.write(f"{rel},{lab}\n")
        else:
            raise ValueError(spec.layout)

    return spec.out_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--layout",
        default="imagefolder",
        choices=["imagefolder", "manifest_csv"],
    )
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
