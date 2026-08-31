import numpy as np
from luminance import generate_solid_angle_weights


def analyze_rgb_energy(img: np.ndarray) -> dict:
    """Measures total RGB energy factoring in equirectangular weighting."""
    h, w, _ = img.shape
    weights = generate_solid_angle_weights(h, w)

    # Apply weights to all channels
    weighted_img = img * weights[..., np.newaxis]

    # Calculate total energy per channel
    total_r = np.sum(weighted_img[..., 0])
    total_g = np.sum(weighted_img[..., 1])
    total_b = np.sum(weighted_img[..., 2])
    total_rgb = total_r + total_g + total_b
    total_rgb = float(total_rgb) if total_rgb > 0 else 1e-6

    return {
        "total_energy_r": float(total_r),
        "total_energy_g": float(total_g),
        "total_energy_b": float(total_b),
        "balance_r_pct": float((total_r / total_rgb) * 100),
        "balance_g_pct": float((total_g / total_rgb) * 100),
        "balance_b_pct": float((total_b / total_rgb) * 100)
    }