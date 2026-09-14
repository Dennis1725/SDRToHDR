import numpy as np

# Reproducible thresholds. These are EXR luminance units unless the source
# EXR is absolutely calibrated. Both images always use the same thresholds.
HIGHLIGHT_THRESHOLD = 1000.0
SHADOW_THRESHOLD = 0.1
HIGHLIGHT_CLIP_THRESHOLD = 10000.0
SHADOW_CLIP_THRESHOLD = 0.01

LUMINANCE_EPSILON = 1e-6


def calculate_luminance(img: np.ndarray) -> np.ndarray:
    """Calculate Rec.709 relative luminance from RGB channels."""
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError("Expected an image array with shape (H, W, 3+).")
    return (
        0.2126 * img[..., 0]
        + 0.7152 * img[..., 1]
        + 0.0722 * img[..., 2]
    )


def generate_solid_angle_weights(height: int, width: int) -> np.ndarray:
    """Normalized sin(theta) weights for equirectangular images."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    theta = np.linspace(0.0, np.pi, height)
    weights_1d = np.sin(theta)
    mean_weight = float(np.mean(weights_1d))
    if mean_weight <= 0.0:
        return np.ones((height, width), dtype=np.float32)

    weights_1d = weights_1d / mean_weight
    return np.broadcast_to(weights_1d[:, np.newaxis], (height, width)).copy()


def _finite_nonnegative_luminance(lum: np.ndarray) -> tuple[np.ndarray, int]:
    """Return finite, non-negative luminance and the invalid-pixel count."""
    arr = np.asarray(lum, dtype=np.float64)
    finite = np.isfinite(arr)
    nonnegative = np.where(finite, np.maximum(arr, 0.0), 0.0)
    invalid_count = int(np.count_nonzero(~finite))
    return nonnegative, invalid_count


def analyze_luminance(img: np.ndarray) -> dict:
    """
    Compute reproducible HDR luminance statistics.

    Important:
    - Statistics are calculated on the original image, before any resizing.
    - Thresholds are fixed and identical for Reference and Generated.
    - Values are reported as EXR units; no automatic cd/m² conversion is made.
    """
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError("Expected an image array with shape (H, W, 3+).")

    h, w, _ = img.shape
    raw_lum = calculate_luminance(img).astype(np.float64, copy=False)
    lum, invalid_count = _finite_nonnegative_luminance(raw_lum)

    if lum.size == 0:
        raise ValueError("Empty luminance data: image has no pixels.")

    positive_lum = lum[lum > 0.0]

    if positive_lum.size == 0:
        min_lum = LUMINANCE_EPSILON
        max_lum = LUMINANCE_EPSILON
        pcts = {
            key: 0.0
            for key in (
                "p0_1", "p1", "p5", "p25", "p50",
                "p75", "p95", "p99", "p99_9"
            )
        }
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

    weights = generate_solid_angle_weights(h, w)
    weighted_lum = lum * weights

    total_pixels = int(lum.size)
    invalid_ratio = invalid_count / total_pixels if total_pixels else 0.0

    highlight_mask = lum > HIGHLIGHT_THRESHOLD
    shadow_mask = lum < SHADOW_THRESHOLD
    highlight_clip_mask = lum > HIGHLIGHT_CLIP_THRESHOLD
    shadow_clip_mask = lum < SHADOW_CLIP_THRESHOLD

    thresholds = {
        "highlight_threshold": HIGHLIGHT_THRESHOLD,
        "shadow_threshold": SHADOW_THRESHOLD,
        "highlight_clip_threshold": HIGHLIGHT_CLIP_THRESHOLD,
        "shadow_clip_threshold": SHADOW_CLIP_THRESHOLD,
        "unit": "EXR units",
        "absolute_cd_m2_calibrated": False,
        "note": (
            "Thresholds are fixed EXR luminance units. "
            "They must not be interpreted as cd/m² unless the source EXR "
            "is absolutely calibrated."
        ),
    }

    return {
        "min_luminance": float(min_lum),
        "max_luminance": float(max_lum),
        "avg_luminance": float(np.mean(weighted_lum)),
        "median_luminance": float(np.median(lum)),
        "dynamic_range_ev": dynamic_range_ev,
        "robust_dynamic_range_ev": robust_dynamic_range_ev,
        "thresholds": thresholds,
        "total_pixels": total_pixels,
        "invalid_pixel_count": invalid_count,
        "invalid_pixel_ratio": float(invalid_ratio),
        "highlight_pixel_count": int(np.count_nonzero(highlight_mask)),
        "shadow_pixel_count": int(np.count_nonzero(shadow_mask)),
        "highlight_clipping_pixel_count": int(np.count_nonzero(highlight_clip_mask)),
        "shadow_clipping_pixel_count": int(np.count_nonzero(shadow_clip_mask)),
        "highlight_pixel_ratio": float(np.mean(highlight_mask)),
        "shadow_pixel_ratio": float(np.mean(shadow_mask)),
        "highlight_clipping_ratio": float(np.mean(highlight_clip_mask)),
        "shadow_clipping_ratio": float(np.mean(shadow_clip_mask)),
        "percentiles": pcts,
    }
