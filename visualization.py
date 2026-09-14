import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from luminance import (
    calculate_luminance,
    LUMINANCE_EPSILON,
    HIGHLIGHT_THRESHOLD,
    SHADOW_THRESHOLD,
    HIGHLIGHT_CLIP_THRESHOLD,
    SHADOW_CLIP_THRESHOLD,
)

_have_skimage = False
_have_ssim = False
try:
    from skimage.color import rgb2lab, deltaE_ciede2000
    from skimage.metrics import structural_similarity as skimage_ssim
    _have_skimage = True
    _have_ssim = True
except Exception:
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        _have_ssim = True
    except Exception:
        pass


def _choose_interpolation(src_shape, dst_shape):
    h_src, w_src = src_shape[:2]
    h_dst, w_dst = dst_shape[:2]
    return cv2.INTER_CUBIC if (h_dst > h_src or w_dst > w_src) else cv2.INTER_AREA


def _match_image_shapes(img_a: np.ndarray, img_b: np.ndarray) -> tuple:
    """Resize only for pixelwise comparison; never for HDR image statistics."""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    if (h_a, w_a) == (h_b, w_b):
        return img_a.copy(), img_b.copy()

    interp_b = _choose_interpolation((h_b, w_b), (h_a, w_a))
    resized_b = cv2.resize(img_b, (w_a, h_a), interpolation=interp_b)
    return img_a.copy(), resized_b


def prepare_pixelwise_pair(img1: np.ndarray, img2: np.ndarray) -> tuple:
    return _match_image_shapes(img1, img2)


def _sanitize_luminance(img: np.ndarray) -> np.ndarray:
    lum = calculate_luminance(img).astype(np.float64, copy=False)
    return np.nan_to_num(
        lum,
        nan=0.0,
        neginf=0.0,
        posinf=np.finfo(np.float64).max,
    ).clip(min=0.0)


def create_false_color_image(
    img: np.ndarray,
    log_vmin: float | None = None,
    log_vmax: float | None = None,
) -> np.ndarray:
    """Create false-color log-luminance using an optionally shared scale."""
    lum = _sanitize_luminance(img)
    log_lum = np.log10(np.maximum(lum, LUMINANCE_EPSILON))

    if log_vmin is None:
        log_vmin = float(np.min(log_lum))
    if log_vmax is None:
        log_vmax = float(np.max(log_lum))

    if log_vmax <= log_vmin:
        lum_norm = np.zeros_like(log_lum)
    else:
        lum_norm = (log_lum - log_vmin) / (log_vmax - log_vmin)
        lum_norm = np.clip(lum_norm, 0.0, 1.0)

    lum_8bit = np.uint8(np.round(lum_norm * 255.0))
    return cv2.applyColorMap(lum_8bit, cv2.COLORMAP_JET)


def _shared_log_range(lum1: np.ndarray, lum2: np.ndarray) -> tuple[float, float]:
    log1 = np.log10(np.maximum(lum1, LUMINANCE_EPSILON))
    log2 = np.log10(np.maximum(lum2, LUMINANCE_EPSILON))
    return (
        float(min(np.min(log1), np.min(log2))),
        float(max(np.max(log1), np.max(log2))),
    )


def save_false_color(img: np.ndarray, output_path: str):
    false_color = create_false_color_image(img)
    cv2.imwrite(output_path, false_color)


def save_histograms(img: np.ndarray, output_path: str):
    """Save RGB log-intensity histogram for a single image."""
    safe = np.nan_to_num(
        img[..., :3],
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    r = np.log10(np.maximum(safe[..., 0].ravel(), LUMINANCE_EPSILON))
    g = np.log10(np.maximum(safe[..., 1].ravel(), LUMINANCE_EPSILON))
    b = np.log10(np.maximum(safe[..., 2].ravel(), LUMINANCE_EPSILON))

    plt.figure(figsize=(10, 6))
    lo = min(float(r.min()), float(g.min()), float(b.min()))
    hi = max(float(r.max()), float(g.max()), float(b.max()))
    bins = np.linspace(lo, hi if hi > lo else lo + 1.0, 100)
    plt.hist(r, bins=bins, alpha=0.5, label="Red (log10)")
    plt.hist(g, bins=bins, alpha=0.5, label="Green (log10)")
    plt.hist(b, bins=bins, alpha=0.5, label="Blue (log10)")
    plt.title("RGB Energy Distribution (Logarithmic)")
    plt.xlabel("Log10(Intensity)")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _wasserstein_1d_from_cdf(pdf1, pdf2, bin_edges):
    cdf1 = np.cumsum(pdf1)
    cdf2 = np.cumsum(pdf2)
    bin_widths = np.diff(bin_edges)
    return float(np.sum(np.abs(cdf1 - cdf2) * bin_widths))


def compute_wasserstein_distance(
    lum1: np.ndarray, lum2: np.ndarray, bins: int = 200
) -> float:
    """Approximate 1-D Wasserstein distance in log10 luminance space."""
    log1 = np.log10(
        np.maximum(np.asarray(lum1, dtype=np.float64).ravel(), LUMINANCE_EPSILON)
    )
    log2 = np.log10(
        np.maximum(np.asarray(lum2, dtype=np.float64).ravel(), LUMINANCE_EPSILON)
    )
    log1 = log1[np.isfinite(log1)]
    log2 = log2[np.isfinite(log2)]

    if log1.size == 0 or log2.size == 0:
        return float("nan")

    lo = min(float(log1.min()), float(log2.min()))
    hi = max(float(log1.max()), float(log2.max()))
    if hi <= lo:
        return 0.0

    bin_edges = np.linspace(lo, hi, bins + 1)
    h1, _ = np.histogram(log1, bins=bin_edges, density=True)
    h2, _ = np.histogram(log2, bins=bin_edges, density=True)
    return _wasserstein_1d_from_cdf(h1, h2, bin_edges)


def _rgb_to_lab_simple(rgb_img: np.ndarray) -> np.ndarray:
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    shape = rgb_img.shape
    flat = rgb_img.reshape(-1, 3)
    xyz = np.dot(flat, M.T)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[:, 0] / Xn, xyz[:, 1] / Yn, xyz[:, 2] / Zn
    eps = 216.0 / 24389.0
    k = 24389.0 / 27.0

    def f(t):
        return np.where(t > eps, np.cbrt(t), (k * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    lab = np.stack(
        [
            116.0 * fy - 16.0,
            500.0 * (fx - fy),
            200.0 * (fy - fz),
        ],
        axis=1,
    )
    return lab.reshape(shape)


def _shared_color_normalization(img1: np.ndarray, img2: np.ndarray):
    """
    Normalize RGB for the supplementary ΔE metric using one shared scale.

    The scale is based on the joint P99.9 RGB value. This prevents a single
    extreme HDR highlight from making almost all other color differences
    numerically invisible. The normalization is ONLY for ΔE visualization/
    color comparison and does not alter HDR luminance statistics.
    """
    rgb1 = np.nan_to_num(
        img1[..., :3].astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rgb2 = np.nan_to_num(
        img2[..., :3].astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rgb1 = np.clip(rgb1, 0.0, None)
    rgb2 = np.clip(rgb2, 0.0, None)

    combined = np.concatenate((rgb1.reshape(-1), rgb2.reshape(-1)))
    finite = combined[np.isfinite(combined)]
    scale = float(np.percentile(finite, 99.9)) if finite.size else 1.0
    scale = max(scale, LUMINANCE_EPSILON)

    return (
        np.clip(rgb1 / scale, 0.0, 1.0),
        np.clip(rgb2 / scale, 0.0, 1.0),
        scale,
    )


def compute_deltaE_map(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute supplementary per-pixel ΔE2000 on shared-normalized RGB.

    This is NOT a direct HDR luminance-fidelity metric. The common normalization
    scale is returned separately by compute_deltaE_context().
    """
    rgb1, rgb2, _ = _shared_color_normalization(img1, img2)

    if _have_skimage:
        lab1 = rgb2lab(rgb1)
        lab2 = rgb2lab(rgb2)
        try:
            return deltaE_ciede2000(lab1, lab2)
        except Exception:
            return np.linalg.norm(lab1 - lab2, axis=2)

    lab1 = _rgb_to_lab_simple(rgb1)
    lab2 = _rgb_to_lab_simple(rgb2)
    return np.linalg.norm(lab1 - lab2, axis=2)


def compute_deltaE_context(img1: np.ndarray, img2: np.ndarray) -> dict:
    _, _, scale = _shared_color_normalization(img1, img2)
    return {
        "method": "shared P99.9 RGB normalization followed by clipping to [0,1]",
        "shared_rgb_scale": float(scale),
        "note": (
            "Supplementary color comparison only. It is not a direct HDR "
            "luminance metric and does not modify HDR statistics."
        ),
    }


def compute_ssim_index(lum1: np.ndarray, lum2: np.ndarray) -> float:
    """Compute SSIM on log10 luminance with a shared normalization range."""
    log1 = np.log10(
        np.maximum(np.asarray(lum1, dtype=np.float64), LUMINANCE_EPSILON)
    )
    log2 = np.log10(
        np.maximum(np.asarray(lum2, dtype=np.float64), LUMINANCE_EPSILON)
    )

    if log1.shape != log2.shape:
        raise ValueError("SSIM requires equal image dimensions. Prepare a pixelwise pair first.")

    combined_min = min(float(np.min(log1)), float(np.min(log2)))
    combined_max = max(float(np.max(log1)), float(np.max(log2)))

    if combined_max <= combined_min:
        return 1.0

    n1 = (log1 - combined_min) / (combined_max - combined_min)
    n2 = (log2 - combined_min) / (combined_max - combined_min)

    if _have_ssim:
        try:
            return float(skimage_ssim(n1, n2, data_range=1.0))
        except Exception:
            pass

    corr = np.corrcoef(n1.ravel(), n2.ravel())[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def _save_map_as_image(
    map_arr: np.ndarray,
    path: str,
    cmap: str = "viridis",
    vmax: float | None = None,
    title: str | None = None,
    vmin: float | None = None,
):
    plt.figure(figsize=(6, 4))
    im = plt.imshow(map_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_comparison_figure(img1, report1, img2, report2, output_path):
    """Create the final HDR comparison dashboard."""
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    # Original images are never modified for their HDR statistics.
    pix1, pix2 = _match_image_shapes(img1, img2)
    lum1 = _sanitize_luminance(pix1)
    lum2 = _sanitize_luminance(pix2)

    log_vmin, log_vmax = _shared_log_range(lum1, lum2)

    fc1 = create_false_color_image(pix1, log_vmin, log_vmax)
    fc2 = create_false_color_image(pix2, log_vmin, log_vmax)
    cv2.imwrite(os.path.join(out_dir, "false_color_ref.png"), fc1)
    cv2.imwrite(os.path.join(out_dir, "false_color_gen.png"), fc2)

    _save_map_as_image(
        np.log10(np.maximum(lum1, LUMINANCE_EPSILON)),
        os.path.join(out_dir, "log_luminance_ref.png"),
        title="Log10 luminance (reference)",
        vmin=log_vmin,
        vmax=log_vmax,
    )
    _save_map_as_image(
        np.log10(np.maximum(lum2, LUMINANCE_EPSILON)),
        os.path.join(out_dir, "log_luminance_gen.png"),
        title="Log10 luminance (generated)",
        vmin=log_vmin,
        vmax=log_vmax,
    )

    abs_diff = np.abs(lum1 - lum2)
    rel_error = abs_diff / np.maximum(lum1, LUMINANCE_EPSILON)

    abs_finite = abs_diff[np.isfinite(abs_diff)]
    rel_finite = rel_error[np.isfinite(rel_error)]

    abs_vmax = (
        float(np.percentile(abs_finite, 99.0))
        if abs_finite.size else 1.0
    )
    rel_vmax = (
        float(np.percentile(rel_finite, 99.0))
        if rel_finite.size else 1.0
    )
    abs_vmax = max(abs_vmax, LUMINANCE_EPSILON)
    rel_vmax = max(rel_vmax, LUMINANCE_EPSILON)

    _save_map_as_image(
        abs_diff,
        os.path.join(out_dir, "luminance_abs_diff.png"),
        cmap="magma",
        vmin=0.0,
        vmax=abs_vmax,
        title=f"Absolute Luminance Difference (display 0–P99={abs_vmax:.3g})",
    )
    _save_map_as_image(
        rel_error,
        os.path.join(out_dir, "luminance_rel_error.png"),
        cmap="magma",
        vmin=0.0,
        vmax=rel_vmax,
        title=f"Relative Luminance Error (display 0–P99={rel_vmax:.3g})",
    )

    # ΔE: statistics use the full map; P99 is display-only.
    try:
        de_map = compute_deltaE_map(pix1, pix2)
        de_finite = de_map[np.isfinite(de_map)]
        mean_de = float(np.mean(de_finite))
        median_de = float(np.median(de_finite))
        p95_de = float(np.percentile(de_finite, 95.0))
        p99_de = float(np.percentile(de_finite, 99.0))
        p99_9_de = float(np.percentile(de_finite, 99.9))
        max_de = float(np.max(de_finite))
        de_vmax = max(float(np.percentile(de_finite, 99.0)), LUMINANCE_EPSILON)
        de_context = compute_deltaE_context(pix1, pix2)
    except Exception:
        de_map = None
        mean_de = median_de = p95_de = p99_de = p99_9_de = max_de = None
        de_vmax = None
        de_context = {}

    mask_ref = lum1 > HIGHLIGHT_THRESHOLD
    mask_gen = lum2 > HIGHLIGHT_THRESHOLD

    cv2.imwrite(
        os.path.join(out_dir, "highlights_ref.png"),
        mask_ref.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        os.path.join(out_dir, "highlights_gen.png"),
        mask_gen.astype(np.uint8) * 255,
    )

    highlight_diff = np.abs(
        mask_ref.astype(np.int8) - mask_gen.astype(np.int8)
    )
    _save_map_as_image(
        highlight_diff,
        os.path.join(out_dir, "highlights_diff.png"),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        title=f"Highlight Mask Difference (> {HIGHLIGHT_THRESHOLD:g} EXR units)",
    )

    try:
        wdist = compute_wasserstein_distance(lum1, lum2)
    except Exception:
        wdist = None

    try:
        ssim_val = compute_ssim_index(lum1, lum2)
    except Exception:
        ssim_val = None

    pcts1 = report1["luminance"].get("percentiles", {})
    pcts2 = report2["luminance"].get("percentiles", {})
    lum1_stats = report1["luminance"]
    lum2_stats = report2["luminance"]

    max1 = float(lum1_stats.get("max_luminance", 0.0))
    max2 = float(lum2_stats.get("max_luminance", 0.0))

    hr_rows = [
        (
            "Highlight Ratio",
            lum1_stats.get("highlight_pixel_ratio", 0.0) * 100.0,
            lum2_stats.get("highlight_pixel_ratio", 0.0) * 100.0,
        ),
        (
            "Shadow Ratio",
            lum1_stats.get("shadow_pixel_ratio", 0.0) * 100.0,
            lum2_stats.get("shadow_pixel_ratio", 0.0) * 100.0,
        ),
        (
            "Highlight Clipping",
            lum1_stats.get("highlight_clipping_ratio", 0.0) * 100.0,
            lum2_stats.get("highlight_clipping_ratio", 0.0) * 100.0,
        ),
        (
            "Shadow Clipping",
            lum1_stats.get("shadow_clipping_ratio", 0.0) * 100.0,
            lum2_stats.get("shadow_clipping_ratio", 0.0) * 100.0,
        ),
    ]

    dr1 = lum1_stats.get("dynamic_range_ev", 0.0)
    dr2 = lum2_stats.get("dynamic_range_ev", 0.0)
    rdr1 = lum1_stats.get("robust_dynamic_range_ev", 0.0)
    rdr2 = lum2_stats.get("robust_dynamic_range_ev", 0.0)

    thresholds = lum1_stats.get("thresholds", {})

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.05, 1.55, 1.15, 1.15],
        hspace=0.45,
        wspace=0.25,
    )

    # Top: images
    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img2 = fig.add_subplot(gs[0, 1])
    ax_img1.imshow(cv2.cvtColor(fc1, cv2.COLOR_BGR2RGB))
    ax_img1.set_title(f"Reference HDR: {report1['filename']}")
    ax_img1.axis("off")
    ax_img2.imshow(cv2.cvtColor(fc2, cv2.COLOR_BGR2RGB))
    ax_img2.set_title(f"Generated HDR: {report2['filename']}")
    ax_img2.axis("off")

    # Left-middle: keep the two tables in separate, non-overlapping axes.
    # Using fixed nested rows is more reliable than placing/scaling both
    # tables inside one axis, because Matplotlib table scaling can otherwise
    # make the first table extend into the second one.
    metrics_gs = gs[1, 0].subgridspec(2, 1, height_ratios=[0.62, 0.38], hspace=0.08)
    ax_hdr = fig.add_subplot(metrics_gs[0, 0])
    ax_pair = fig.add_subplot(metrics_gs[1, 0])
    ax_hdr.axis("off")
    ax_pair.axis("off")

    hdr_data = [
        ["HDR Luminance Metric", "Reference", "Generated"],
        ["Dynamic Range (EV)", f"{dr1:.2f}", f"{dr2:.2f}"],
        ["Robust Dynamic Range (EV)", f"{rdr1:.2f}", f"{rdr2:.2f}"],
        ["Maximum Luminance", f"{max1:.3f}", f"{max2:.3f}"],
        ["Median Luminance", f"{lum1_stats.get('median_luminance', 0.0):.3f}",
         f"{lum2_stats.get('median_luminance', 0.0):.3f}"],
        ["Highlight Ratio", f"{hr_rows[0][1]:.4f} %", f"{hr_rows[0][2]:.4f} %"],
        ["Shadow Ratio", f"{hr_rows[1][1]:.4f} %", f"{hr_rows[1][2]:.4f} %"],
        ["Highlight Clipping", f"{hr_rows[2][1]:.4f} %", f"{hr_rows[2][2]:.4f} %"],
        ["Shadow Clipping", f"{hr_rows[3][1]:.4f} %", f"{hr_rows[3][2]:.4f} %"],
        ["Highlight Pixels", str(lum1_stats.get("highlight_pixel_count", 0)),
         str(lum2_stats.get("highlight_pixel_count", 0))],
        ["Shadow Pixels", str(lum1_stats.get("shadow_pixel_count", 0)),
         str(lum2_stats.get("shadow_pixel_count", 0))],
    ]

    table = ax_hdr.table(
        cellText=hdr_data,
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
        cellLoc="left",
        colWidths=[0.54, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)

    # Pairwise metrics are deliberately separate and have their own axis.
    pairwise_data = [
        ["Pairwise Metric", "Value", "Direction"],
        ["SSIM (log10 luminance)", f"{ssim_val:.4f}" if ssim_val is not None else "n/a", "higher = better"],
        ["Wasserstein (log10 luminance)", f"{wdist:.4f}" if wdist is not None else "n/a", "lower = better"],
        ["Mean ΔE2000", f"{mean_de:.3f}" if mean_de is not None else "n/a", "lower = better"],
        ["Median ΔE2000", f"{median_de:.3f}" if median_de is not None else "n/a", "lower = better"],
        ["P95 ΔE2000", f"{p95_de:.3f}" if p95_de is not None else "n/a", "lower = better"],
        ["P99 ΔE2000", f"{p99_de:.3f}" if p99_de is not None else "n/a", "lower = better"],
        ["Max ΔE2000", f"{max_de:.3f}" if max_de is not None else "n/a", "lower = better"],
    ]

    pair_table = ax_pair.table(
        cellText=pairwise_data,
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
        cellLoc="left",
        colWidths=[0.54, 0.22, 0.22],
    )
    pair_table.auto_set_font_size(False)
    pair_table.set_fontsize(7.6)

    # Percentiles
    ax_pct = fig.add_subplot(gs[1, 1])
    keys = [
        ("p1", "P1"),
        ("p5", "P5"),
        ("p50", "P50"),
        ("p95", "P95"),
        ("p99", "P99"),
        ("p99_9", "P99.9"),
    ]
    x = np.arange(len(keys))
    vals1 = [pcts1.get(k, 0.0) for k, _ in keys]
    vals2 = [pcts2.get(k, 0.0) for k, _ in keys]
    width = 0.35

    ax_pct.bar(x - width / 2, vals1, width, label="Reference")
    ax_pct.bar(x + width / 2, vals2, width, label="Generated")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels([label for _, label in keys])
    ax_pct.set_ylabel("Luminance (EXR units)")
    ax_pct.set_title("Luminance Percentiles")
    ax_pct.grid(True, axis="y", alpha=0.3)
    ax_pct.legend(fontsize=8)

    # Difference maps
    ax_diff = fig.add_subplot(gs[2, 0])
    ax_diff.axis("off")
    ax_diff.set_title("Difference Maps", fontsize=11)

    sub_diff = gs[2, 0].subgridspec(1, 2, wspace=0.12)
    ax_abs = fig.add_subplot(sub_diff[0, 0])
    ax_rel = fig.add_subplot(sub_diff[0, 1])

    im_abs = ax_abs.imshow(abs_diff, cmap="magma", vmin=0, vmax=abs_vmax)
    ax_abs.set_title("Absolute Luminance Difference")
    ax_abs.axis("off")
    fig.colorbar(im_abs, ax=ax_abs, fraction=0.046, pad=0.02)

    im_rel = ax_rel.imshow(rel_error, cmap="magma", vmin=0, vmax=rel_vmax)
    ax_rel.set_title("Relative Luminance Error")
    ax_rel.axis("off")
    fig.colorbar(im_rel, ax=ax_rel, fraction=0.046, pad=0.02)

    # Histogram
    ax_hist = fig.add_subplot(gs[2, 1])
    log1 = np.log10(np.maximum(lum1.ravel(), LUMINANCE_EPSILON))
    log2 = np.log10(np.maximum(lum2.ravel(), LUMINANCE_EPSILON))
    bins = np.linspace(
        log_vmin,
        log_vmax if log_vmax > log_vmin else log_vmin + 1.0,
        80,
    )
    ax_hist.hist(log1, bins=bins, alpha=0.55, label="Reference")
    ax_hist.hist(log2, bins=bins, alpha=0.55, label="Generated")
    ax_hist.set_title("Log10 Luminance Distribution")
    ax_hist.set_xlabel("log10(luminance)")
    ax_hist.set_ylabel("Pixel count")
    ax_hist.legend(fontsize=8)

    # Highlight masks
    ax_hl = fig.add_subplot(gs[3, 0])
    ax_hl.axis("off")
    ax_hl.set_title("Highlight Region Analysis", fontsize=11)

    sub_hl = gs[3, 0].subgridspec(1, 2, wspace=0.12)
    ah1 = fig.add_subplot(sub_hl[0, 0])
    ah2 = fig.add_subplot(sub_hl[0, 1])

    ah1.imshow(mask_ref, cmap="gray", vmin=0, vmax=1)
    ah1.set_title(
        f"Reference > {HIGHLIGHT_THRESHOLD:g}\n"
        f"{np.count_nonzero(mask_ref)} pixels"
    )
    ah1.axis("off")

    ah2.imshow(mask_gen, cmap="gray", vmin=0, vmax=1)
    ah2.set_title(
        f"Generated > {HIGHLIGHT_THRESHOLD:g}\n"
        f"{np.count_nonzero(mask_gen)} pixels"
    )
    ah2.axis("off")

    # ΔE
    ax_de = fig.add_subplot(gs[3, 1])
    if de_map is not None:
        im_de = ax_de.imshow(
            de_map,
            cmap="inferno",
            vmin=0,
            vmax=de_vmax,
        )
        ax_de.set_title(
            f"ΔE2000 Heatmap (display 0–P99={de_vmax:.2f})"
        )
        ax_de.axis("off")
        fig.colorbar(im_de, ax=ax_de, fraction=0.046, pad=0.02)

        stats_text = (
            f"Mean: {mean_de:.3f}\n"
            f"Median: {median_de:.3f}\n"
            f"P95: {p95_de:.3f}\n"
            f"P99: {p99_de:.3f}\n"
            f"P99.9: {p99_9_de:.3f}\n"
            f"Max: {max_de:.3f}\n"
            f"Shared RGB scale: {de_context.get('shared_rgb_scale', float('nan')):.4g}\n"
            "* supplementary color metric"
        )
        ax_de.text(
            0.01, 0.99, stats_text,
            va="top", ha="left",
            fontsize=7.5,
            family="monospace",
            transform=ax_de.transAxes,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
            ),
        )
    else:
        ax_de.axis("off")
        ax_de.text(
            0.5, 0.5,
            "ΔE unavailable",
            ha="center",
            va="center",
        )

    unit = thresholds.get("unit", "EXR units")

    fig.suptitle("HDR Comparison", fontsize=16, y=0.995)
    fig.text(
        0.5, 0.005,
        f"Thresholds: Highlight > {HIGHLIGHT_THRESHOLD:g} {unit} | "
        f"Shadow < {SHADOW_THRESHOLD:g} {unit} | "
        f"Highlight clipping > {HIGHLIGHT_CLIP_THRESHOLD:g} {unit} | "
        f"Shadow clipping < {SHADOW_CLIP_THRESHOLD:g} {unit}\n"
        "Absolute cd/m² interpretation requires an absolutely calibrated source EXR.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)