import numpy as np

# Fixed, identical thresholds for Reference and Generated.
# Values are EXR luminance units, not automatically cd/m².
HIGHLIGHT_THRESHOLD = 1000.0
SHADOW_THRESHOLD = 0.1
HIGHLIGHT_CLIP_THRESHOLD = 10000.0
SHADOW_CLIP_THRESHOLD = 0.01

LUMINANCE_EPSILON = 1e-6


def calculate_luminance(img: np.ndarray) -> np.ndarray:
    """Calculate relative luminance using Rec. 709 coefficients."""
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError("Expected an image array with shape (H, W, 3+).")
    return (
        0.2126 * img[..., 0]
        + 0.7152 * img[..., 1]
        + 0.0722 * img[..., 2]
    )


def generate_solid_angle_weights(height: int, width: int) -> np.ndarray:
    """Generate normalized solid-angle weights for an equirectangular image."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    theta = np.linspace(0.0, np.pi, height)
    weights_1d = np.sin(theta)
    mean_weight = float(np.mean(weights_1d))
    if mean_weight <= 0.0:
        return np.ones((height, width), dtype=np.float32)

    weights_1d = weights_1d / mean_weight
    return np.broadcast_to(weights_1d[:, np.newaxis], (height, width)).copy()


def _sanitize_luminance(lum: np.ndarray) -> np.ndarray:
    """Return finite, non-negative luminance for statistical analysis."""
    lum = np.asarray(lum, dtype=np.float64)
    return np.nan_to_num(
        lum,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    ).clip(min=0.0)


def analyze_luminance(img: np.ndarray) -> dict:
    """Return HDR luminance statistics and fixed-threshold region metrics."""
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError("Expected an image array with shape (H, W, 3+).")

    h, w, _ = img.shape
    lum = _sanitize_luminance(calculate_luminance(img))
    if lum.size == 0:
        raise ValueError("Empty luminance data: image has no pixels.")

    positive_lum = lum[np.isfinite(lum) & (lum > 0.0)]
    if positive_lum.size == 0:
        min_lum = LUMINANCE_EPSILON
        max_lum = LUMINANCE_EPSILON
        pcts = {key: 0.0 for key in (
            "p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9"
        )}
    else:
        min_lum = float(np.min(positive_lum))
        max_lum = float(np.max(positive_lum))
        pcts = {
            "p0_1": float(np.percentile(positive_lum, 0.1)),
            "p1": float(np.percentile(positive_lum, 1.0)),
            "p5": float(np.percentile(positive_lum, 5.0)),
            "p25": float(np.percentile(positive_lum, 25.0)),
            "p50": float(np.percentile(positive_lum, 50.0)),
            "p75": float(np.percentile(positive_lum, 75.0)),
            "p95": float(np.percentile(positive_lum, 95.0)),
            "p99": float(np.percentile(positive_lum, 99.0)),
            "p99_9": float(np.percentile(positive_lum, 99.9)),
        }

    dynamic_range_ev = (
        float(np.log2(max_lum / min_lum))
        if max_lum > min_lum > 0.0
        else 0.0
    )

    p0_1 = pcts["p0_1"]
    p99_9 = pcts["p99_9"]
    robust_dynamic_range_ev = (
        float(np.log2(p99_9 / p0_1))
        if p99_9 > p0_1 > 0.0
        else 0.0
    )

    # Equirectangular weighting is used only for the mean luminance.
    weights = generate_solid_angle_weights(h, w)
    weighted_lum = lum * weights

    thresholds = {
        "highlight_threshold": HIGHLIGHT_THRESHOLD,
        "shadow_threshold": SHADOW_THRESHOLD,
        "highlight_clip_threshold": HIGHLIGHT_CLIP_THRESHOLD,
        "shadow_clip_threshold": SHADOW_CLIP_THRESHOLD,
        "unit": "EXR units",
        "absolute_cd_m2_calibrated": False,
    }

    return {
        "min_luminance": float(min_lum),
        "max_luminance": float(max_lum),
        "avg_luminance": float(np.mean(weighted_lum)),
        "median_luminance": float(np.median(lum)),
        "dynamic_range_ev": dynamic_range_ev,
        "robust_dynamic_range_ev": robust_dynamic_range_ev,
        "thresholds": thresholds,
        "highlight_pixel_ratio": float(np.mean(lum > HIGHLIGHT_THRESHOLD)),
        "shadow_pixel_ratio": float(np.mean(lum < SHADOW_THRESHOLD)),
        "highlight_clipping_ratio": float(np.mean(lum > HIGHLIGHT_CLIP_THRESHOLD)),
        "shadow_clipping_ratio": float(np.mean(lum < SHADOW_CLIP_THRESHOLD)),
        "percentiles": pcts,
    }
