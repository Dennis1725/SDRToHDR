import numpy as np
from luminance import generate_solid_angle_weights


def analyze_rgb_energy(img: np.ndarray) -> dict:
    """Measure RGB energy using equirectangular solid-angle weighting."""
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError("Expected an image array with shape (H, W, 3+).")

    h, w, _ = img.shape
    weights = generate_solid_angle_weights(h, w)
    safe_img = np.nan_to_num(
        img[..., :3].astype(np.float64, copy=False),
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    weighted_img = safe_img * weights[..., np.newaxis]

    total_r = float(np.sum(weighted_img[..., 0]))
    total_g = float(np.sum(weighted_img[..., 1]))
    total_b = float(np.sum(weighted_img[..., 2]))
    total_rgb = total_r + total_g + total_b
    denominator = total_rgb if total_rgb > 0.0 else 1e-6

    return {
        "total_energy_r": total_r,
        "total_energy_g": total_g,
        "total_energy_b": total_b,
        "balance_r_pct": float((total_r / denominator) * 100.0),
        "balance_g_pct": float((total_g / denominator) * 100.0),
        "balance_b_pct": float((total_b / denominator) * 100.0),
    }
