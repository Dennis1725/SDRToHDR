import numpy as np


def calculate_luminance(img: np.ndarray) -> np.ndarray:
    """Calculates relative luminance using standard Rec. 709 coefficients."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def generate_solid_angle_weights(height: int, width: int) -> np.ndarray:
    """
    Generates a weight map for equirectangular distortion.
    Pixels at the poles (zenith/nadir) cover less spherical area than at the equator.
    """
    # Theta represents the vertical angle from 0 (top) to pi (bottom)
    theta = np.linspace(0, np.pi, height)
    weights = np.sin(theta)
    mean_weight = np.mean(weights)
    if mean_weight <= 0:
        return np.ones((height, width), dtype=np.float32)

    # Normalize weights so their average is 1.0
    weights = weights / mean_weight

    # Broadcast to match image dimensions
    return np.tile(weights[:, np.newaxis], (1, width))


def analyze_luminance(img: np.ndarray) -> dict:
    """Returns minimum, maximum, average, and EV dynamic range."""
    h, w, _ = img.shape
    lum = calculate_luminance(img)
    weights = generate_solid_angle_weights(h, w)

    weighted_lum = lum * weights

    # Avoid log(0) for pitch black pixels
    min_lum = np.min(lum[lum > 1e-6]) if np.any(lum > 1e-6) else 1e-6
    max_lum = np.max(lum)

    # Dynamic range in Exposure Values (Stops)
    ev_range = np.log2(max_lum / min_lum) if max_lum > min_lum else 0.0

    return {
        "min_luminance": float(min_lum),
        "max_luminance": float(max_lum),
        "avg_luminance": float(np.mean(weighted_lum)),
        "median_luminance": float(np.median(lum)),
        "dynamic_range_ev": float(ev_range)
    }