from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int, int], sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=sigma)
    noise -= float(noise.mean())
    std = float(noise.std())
    if std > 1e-6:
        noise /= std
    return noise


def keep_largest_component(volume: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(volume)
    if count <= 1:
        return volume.astype(bool)
    sizes = ndimage.sum(volume, labeled, index=np.arange(1, count + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


def gaussian_blob(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    weight: float = 1.0,
) -> np.ndarray:
    cz, cy, cx = center
    rz, ry, rx = radii
    return weight * np.exp(-0.5 * (((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2))


def make_shapr_like_volume(
    rng: np.random.Generator,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    kind: str,
) -> np.ndarray:
    field = np.zeros_like(z, dtype=np.float32)

    if kind == "shapr_spiky_disc":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.16, 0.38, 0.38), 0.95)
        field -= gaussian_blob(z, y, x, (0.02, 0.02, -0.02), (0.10, 0.16, 0.16), 0.24)

        for i, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 11, endpoint=False)):
            angle += rng.uniform(-0.10, 0.10)
            length = rng.uniform(0.18, 0.34)
            width = rng.uniform(0.055, 0.090)
            for step, t in enumerate((0.46, 0.56, 0.66)):
                r = t + length * step / 3.0
                cy = np.sin(angle) * r
                cx = np.cos(angle) * r
                field += gaussian_blob(
                    z,
                    y,
                    x,
                    (rng.uniform(-0.06, 0.08), cy, cx),
                    (rng.uniform(0.07, 0.12), width, width),
                    0.28 - 0.045 * step,
                )

        for angle in np.linspace(np.pi / 11, 2.0 * np.pi + np.pi / 11, 5, endpoint=False):
            field -= gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.04, 0.08), np.sin(angle) * 0.42, np.cos(angle) * 0.42),
                (0.09, 0.10, 0.10),
                0.12,
            )
        threshold = 0.34
    elif kind == "shapr_branching":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.23, 0.32, 0.34), 0.86)
        arms = [0.2, 1.65, 2.85, 4.25, 5.35]
        for idx, angle in enumerate(arms):
            angle += rng.uniform(-0.14, 0.14)
            length = rng.uniform(0.36, 0.55)
            arm_width = rng.uniform(0.10, 0.15)
            for step, t in enumerate(np.linspace(0.22, length, 4)):
                bend = 0.10 * np.sin(step + idx)
                cy = np.sin(angle + bend) * t
                cx = np.cos(angle + bend) * t
                field += gaussian_blob(
                    z,
                    y,
                    x,
                    (rng.uniform(-0.09, 0.10), cy, cx),
                    (rng.uniform(0.11, 0.17), arm_width * (1.12 - 0.12 * step), arm_width * (1.12 - 0.12 * step)),
                    0.45 - 0.045 * step,
                )
            tip_r = length + 0.08
            field += gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.10, 0.10), np.sin(angle) * tip_r, np.cos(angle) * tip_r),
                (0.09, arm_width * 0.85, arm_width * 0.85),
                0.22,
            )
        field -= gaussian_blob(z, y, x, (0.08, -0.10, 0.16), (0.12, 0.13, 0.13), 0.16)
        threshold = 0.40
    elif kind == "shapr_concave_spikes":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.25, 0.40, 0.39), 0.86)
        for angle in (0.45, 2.15, 3.75, 5.10):
            field += gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.08, 0.08), np.sin(angle) * 0.34, np.cos(angle) * 0.34),
                (0.14, 0.20, 0.16),
                0.42,
            )
        for angle in (1.25, 2.75, 4.55, 5.85):
            field -= gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.06, 0.10), np.sin(angle) * 0.34, np.cos(angle) * 0.34),
                (0.13, 0.16, 0.16),
                0.24,
            )
        for angle in (0.0, 2.95, 4.90):
            field += gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.07, 0.09), np.sin(angle) * 0.58, np.cos(angle) * 0.58),
                (0.08, 0.08, 0.08),
                0.18,
            )
        field -= gaussian_blob(z, y, x, (0.04, 0.02, 0.02), (0.11, 0.18, 0.18), 0.18)
        threshold = 0.39
    elif kind == "shapr_disc":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.16, 0.44, 0.45), 1.02)
        field += gaussian_blob(z, y, x, (-0.07, -0.10, -0.13), (0.13, 0.20, 0.20), 0.34)
        field += gaussian_blob(z, y, x, (0.08, 0.15, 0.16), (0.12, 0.20, 0.18), 0.28)
        field -= gaussian_blob(z, y, x, (0.03, 0.01, 0.02), (0.12, 0.18, 0.18), 0.36)
        threshold = 0.36
    elif kind == "shapr_lobed":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.26, 0.36, 0.36), 0.76)
        for angle in np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False):
            angle += rng.uniform(-0.16, 0.16)
            r = rng.uniform(0.24, 0.38)
            cy = np.sin(angle) * r
            cx = np.cos(angle) * r
            cz = rng.uniform(-0.10, 0.10)
            field += gaussian_blob(
                z,
                y,
                x,
                (cz, cy, cx),
                (rng.uniform(0.13, 0.20), rng.uniform(0.14, 0.21), rng.uniform(0.14, 0.21)),
                rng.uniform(0.48, 0.72),
            )
        field -= gaussian_blob(z, y, x, (0.08, -0.12, 0.18), (0.15, 0.15, 0.15), 0.24)
        field += gaussian_blob(z, y, x, (-0.11, 0.05, -0.22), (0.12, 0.13, 0.16), 0.20)
        threshold = 0.44
    elif kind == "shapr_star":
        field += gaussian_blob(z, y, x, (0.00, 0.00, 0.00), (0.24, 0.31, 0.31), 0.82)
        for angle in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False):
            angle += rng.uniform(-0.11, 0.11)
            r = rng.uniform(0.31, 0.46)
            cy = np.sin(angle) * r
            cx = np.cos(angle) * r
            field += gaussian_blob(
                z,
                y,
                x,
                (rng.uniform(-0.10, 0.12), cy, cx),
                (rng.uniform(0.10, 0.16), rng.uniform(0.12, 0.18), rng.uniform(0.12, 0.18)),
                rng.uniform(0.52, 0.76),
            )
        for angle in np.linspace(np.pi / 6, 2.0 * np.pi + np.pi / 6, 3, endpoint=False):
            field -= gaussian_blob(
                z,
                y,
                x,
                (0.03, np.sin(angle) * 0.33, np.cos(angle) * 0.33),
                (0.13, 0.12, 0.12),
                0.20,
            )
        field += gaussian_blob(z, y, x, (-0.12, -0.16, 0.18), (0.10, 0.14, 0.13), 0.22)
        threshold = 0.43
    else:
        raise ValueError(f"Unsupported SHAPR-like kind: {kind}")

    for _ in range(4):
        center = tuple(rng.uniform(-0.24, 0.24, size=3).astype(np.float32))
        radii = tuple(rng.uniform(0.10, 0.22, size=3).astype(np.float32))
        field += gaussian_blob(z, y, x, center, radii, rng.uniform(0.05, 0.12))
    for _ in range(2):
        center = tuple(rng.uniform(-0.26, 0.26, size=3).astype(np.float32))
        radii = tuple(rng.uniform(0.10, 0.20, size=3).astype(np.float32))
        field -= gaussian_blob(z, y, x, center, radii, rng.uniform(0.04, 0.10))

    field += 0.045 * smooth_noise(rng, z.shape, sigma=3.0)
    field = ndimage.gaussian_filter(field, sigma=0.7)
    volume = field > threshold
    volume = keep_largest_component(volume)
    volume = ndimage.binary_closing(volume, iterations=2)
    volume = ndimage.binary_fill_holes(volume)
    volume = keep_largest_component(volume)
    return volume.astype(np.float32)


def make_cell_volume(
    seed: int,
    resolution: int,
    kind: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")

    if kind.startswith("shapr_"):
        return make_shapr_like_volume(rng, z, y, x, kind)

    if kind == "thorn_disc":
        theta = np.arctan2(y, x)
        r_xy = np.sqrt(x * x + y * y)
        waviness = 0.035 * np.sin(7 * theta + 0.6) + 0.025 * np.sin(11 * theta - 0.4)
        radius = 0.56 + waviness
        thickness = 0.15 + 0.045 * (1.0 - np.clip(r_xy / 0.70, 0.0, 1.0))
        volume = (r_xy <= radius) & (np.abs(z) <= thickness)

        for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
            angle += rng.uniform(-0.08, 0.08)
            angular = np.abs(np.angle(np.exp(1j * (theta - angle))))
            spike_len = rng.uniform(0.12, 0.22)
            spike_width = rng.uniform(0.055, 0.085)
            spike_tip = radius + spike_len * (1.0 - np.clip(angular / spike_width, 0.0, 1.0))
            spike = (angular < spike_width) & (r_xy > radius - 0.03) & (r_xy < spike_tip)
            spike &= np.abs(z) < (0.09 + 0.03 * (1.0 - angular / spike_width))
            volume |= spike

        dimple = (r_xy < 0.22) & (np.abs(z) < 0.055)
        volume &= ~dimple
        dorsal_ridge = ((z + 0.13) / 0.08) ** 2 + ((y - 0.18) / 0.18) ** 2 + ((x + 0.12) / 0.28) ** 2 <= 1.0
        ventral_notch = ((z - 0.13) / 0.08) ** 2 + ((y + 0.22) / 0.20) ** 2 + ((x - 0.16) / 0.22) ** 2 <= 1.0
        volume |= dorsal_ridge
        volume &= ~ventral_notch
        volume = ndimage.binary_closing(volume, iterations=1)
        volume = ndimage.binary_fill_holes(volume)
        volume = keep_largest_component(volume)
        return volume.astype(np.float32)

    if kind == "star":
        theta = np.arctan2(y, x)
        r_xy = np.sqrt((x / 0.95) ** 2 + (y / 0.95) ** 2)
        star_wave = np.maximum(0.0, np.cos(5 * theta + 0.25)) ** 1.8
        radius = 0.38 + 0.28 * star_wave + 0.035 * np.sin(3 * theta)
        thickness = 0.23 + 0.08 * np.maximum(0.0, 1.0 - r_xy / 0.72)
        volume = (r_xy <= radius) & (np.abs(z) <= thickness)

        core = (x / 0.33) ** 2 + (y / 0.33) ** 2 + (z / 0.30) ** 2 <= 1.0
        volume |= core
        for angle in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
            cx = np.cos(angle + 0.25) * 0.42
            cy = np.sin(angle + 0.25) * 0.42
            ray = ((x - cx) / 0.23) ** 2 + ((y - cy) / 0.13) ** 2 + (z / 0.18) ** 2 <= 1.0
            volume |= ray

        dorsal_bulge = ((z + 0.12) / 0.12) ** 2 + ((y + 0.20) / 0.22) ** 2 + ((x - 0.24) / 0.22) ** 2 <= 1.0
        ventral_scoop = ((z - 0.15) / 0.11) ** 2 + ((y - 0.26) / 0.18) ** 2 + ((x + 0.18) / 0.18) ** 2 <= 1.0
        volume |= dorsal_bulge
        volume &= ~ventral_scoop

        volume = ndimage.binary_closing(volume, iterations=1)
        volume = ndimage.binary_fill_holes(volume)
        volume = keep_largest_component(volume)
        return volume.astype(np.float32)

    if kind == "concave_lobed":
        main = (z / 0.42) ** 2 + (y / 0.62) ** 2 + (x / 0.50) ** 2 <= 1.0
        left_lobe = (z / 0.32) ** 2 + ((y + 0.08) / 0.40) ** 2 + ((x + 0.38) / 0.30) ** 2 <= 1.0
        right_lobe = ((z - 0.04) / 0.30) ** 2 + ((y - 0.07) / 0.38) ** 2 + ((x - 0.42) / 0.28) ** 2 <= 1.0
        lower_lobe = ((z + 0.02) / 0.26) ** 2 + ((y - 0.42) / 0.26) ** 2 + (x / 0.32) ** 2 <= 1.0
        volume = main | left_lobe | right_lobe | lower_lobe

        cuts = [
            ((z / 0.32) ** 2 + ((y + 0.40) / 0.24) ** 2 + (x / 0.34) ** 2 <= 1.0),
            (((z - 0.05) / 0.26) ** 2 + ((y - 0.08) / 0.22) ** 2 + ((x + 0.06) / 0.20) ** 2 <= 1.0),
            (((z + 0.02) / 0.26) ** 2 + ((y + 0.02) / 0.22) ** 2 + ((x - 0.22) / 0.18) ** 2 <= 1.0),
        ]
        for cut in cuts:
            volume &= ~cut

        bridge = (z / 0.23) ** 2 + (y / 0.34) ** 2 + (x / 0.42) ** 2 <= 1.0
        volume |= bridge
        boundary_noise = smooth_noise(rng, (resolution, resolution, resolution), sigma=4.5)
        volume &= boundary_noise > -2.1
        volume = ndimage.binary_closing(volume, iterations=2)
        volume = ndimage.binary_fill_holes(volume)
        volume = keep_largest_component(volume)
        return volume.astype(np.float32)

    if kind == "elongated":
        radii = np.array([0.48, 0.78, 0.36], dtype=np.float32)
        roughness = 0.08
        protrusions = 1
        indentations = 0
        smooth_sigma = 5.0
    elif kind == "lobed":
        radii = np.array([0.55, 0.70, 0.48], dtype=np.float32)
        roughness = 0.12
        protrusions = 3
        indentations = 1
        smooth_sigma = 4.5
    elif kind == "spiky":
        radii = np.array([0.50, 0.68, 0.42], dtype=np.float32)
        roughness = 0.18
        protrusions = 5
        indentations = 2
        smooth_sigma = 3.8
    elif kind == "budding":
        radii = np.array([0.52, 0.72, 0.40], dtype=np.float32)
        roughness = 0.09
        protrusions = 2
        indentations = 0
        smooth_sigma = 5.0
    elif kind == "irregular":
        radii = np.array([0.62, 0.69, 0.58], dtype=np.float32)
        roughness = 0.16
        protrusions = 4
        indentations = 2
        smooth_sigma = 4.2
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    radii *= rng.uniform(0.92, 1.08, size=3).astype(np.float32)
    field = (z / radii[0]) ** 2 + (y / radii[1]) ** 2 + (x / radii[2]) ** 2
    boundary_noise = smooth_noise(rng, (resolution, resolution, resolution), sigma=smooth_sigma)
    volume = field <= (1.0 + roughness * boundary_noise)

    for _ in range(protrusions):
        direction = rng.normal(0.0, 1.0, 3)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        center = direction * rng.uniform(0.32, 0.56)
        if kind == "budding":
            blob_r = rng.uniform(0.18, 0.28)
        else:
            blob_r = rng.uniform(0.10, 0.22)
        blob = (
            ((z - center[0]) / blob_r) ** 2
            + ((y - center[1]) / (blob_r * rng.uniform(0.9, 1.35))) ** 2
            + ((x - center[2]) / (blob_r * rng.uniform(0.9, 1.35))) ** 2
        ) <= 1.0
        volume |= blob

    for _ in range(indentations):
        direction = rng.normal(0.0, 1.0, 3)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        center = direction * rng.uniform(0.50, 0.76)
        cut_r = rng.uniform(0.12, 0.22)
        cut = (
            ((z - center[0]) / cut_r) ** 2
            + ((y - center[1]) / (cut_r * rng.uniform(0.8, 1.4))) ** 2
            + ((x - center[2]) / (cut_r * rng.uniform(0.8, 1.4))) ** 2
        ) <= 1.0
        volume &= ~cut

    volume = ndimage.binary_fill_holes(volume)
    volume = ndimage.binary_closing(volume, iterations=1)
    if kind in {"spiky", "irregular"}:
        volume = ndimage.binary_closing(volume, iterations=1)
    else:
        volume = ndimage.binary_opening(volume, iterations=1)
    volume = keep_largest_component(volume)
    volume = ndimage.binary_fill_holes(volume)
    return volume.astype(np.float32)


def extract_views(volume: np.ndarray) -> dict[str, np.ndarray]:
    binary = (volume > 0).astype(np.float32)
    depth, height, _ = binary.shape
    z_mid = depth // 2
    return {
        "top": binary[:z_mid].sum(axis=0) / max(z_mid, 1),
        "bottom": binary[z_mid:].sum(axis=0) / max(depth - z_mid, 1),
        "side": binary.sum(axis=1) / max(height, 1),
    }


def projection_to_image(array: np.ndarray, image_size: int, gamma: float, rng: np.random.Generator | None = None) -> Image.Image:
    array = np.clip(array.astype(np.float32), 0.0, 1.0)
    array = ndimage.gaussian_filter(array, sigma=1.05)
    max_val = float(array.max())
    if max_val > 0:
        array /= max_val
    array = np.power(array, gamma)
    if rng is not None:
        low_freq = smooth_noise(rng, array.shape, sigma=4.0)
        array = np.clip(array + 0.018 * low_freq, 0.0, 1.0)
    img_arr = np.clip(array * 216.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    if image_size != img.width:
        img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return img


def save_sample(out_dir: Path, sample_name: str, volume: np.ndarray, image_size: int, gamma: float) -> None:
    sample_dir = out_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    views = extract_views(volume)
    seed = sum(ord(ch) for ch in sample_name)
    rng = np.random.default_rng(seed)
    images = {name: projection_to_image(view, image_size, gamma, rng) for name, view in views.items()}
    for name, img in images.items():
        img.save(sample_dir / f"{name}.png")
        np.save(sample_dir / f"{name}.npy", views[name].astype(np.float32))
    np.save(sample_dir / "volume.npy", volume.astype(np.float32))

    label_h = 24
    sheet = Image.new("RGB", (image_size * 3 + 4, image_size + label_h + 2), (12, 12, 18))
    draw = ImageDraw.Draw(sheet)
    x = 0
    for name in ("top", "bottom", "side"):
        tile = ImageOps.expand(images[name].convert("RGB"), border=1, fill=(80, 80, 90))
        draw.text((x + 8, 6), name, fill=(220, 225, 235))
        sheet.paste(tile, (x, label_h))
        x += image_size + 2
    sheet.save(sample_dir / "preview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cell-like top/bottom/side images for TriView3D Generator.")
    parser.add_argument("--out", default="synthetic_cells", help="Output folder")
    parser.add_argument("--count", type=int, default=3, help="Number of samples")
    parser.add_argument("--resolution", type=int, default=64, help="Internal volume resolution")
    parser.add_argument("--image-size", type=int, default=64, help="Saved PNG size")
    parser.add_argument("--gamma", type=float, default=0.9, help="Projection contrast gamma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--kinds",
        default="shapr_spiky_disc,shapr_branching,shapr_concave_spikes",
        help="Comma-separated morphology kinds to cycle through",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    if not kinds:
        raise ValueError("At least one kind is required")

    for idx in range(args.count):
        kind = kinds[idx % len(kinds)]
        sample_name = f"{idx + 1:02d}_{kind}"
        volume = make_cell_volume(seed=args.seed + idx, resolution=args.resolution, kind=kind)
        save_sample(out_dir, sample_name, volume, args.image_size, args.gamma)
        print(f"Saved {out_dir / sample_name}")

    print("Done. Upload top.png, bottom.png, side.png from any sample folder in Generator.")


if __name__ == "__main__":
    main()
